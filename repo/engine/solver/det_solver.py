"""
RT-DETRv4: Painlessly Furthering Real-Time Object Detection with Vision Foundation Models
Copyright (c) 2025 The RT-DETRv4 Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
"""

import time
import json
import datetime
import math
from pathlib import Path

import torch

from ..misc import dist_utils, stats

from ._solver import BaseSolver
from .det_engine import train_one_epoch, evaluate
from ..optim.lr_scheduler import FlatCosineLRScheduler


def _early_stopping_score(stats, monitor):
    """Return a scalar validation score for the configured monitor."""
    metric_indexes = {
        'chem_box_precision': ('chem_eval_box', 0),
        'chem_box_recall': ('chem_eval_box', 1),
        'chem_box_f1': ('chem_eval_box', 2),
        'chem_box_iou': ('chem_eval_box', 3),
        'chem_line_precision': ('chem_eval_line_v2', 0),
        'chem_line_recall': ('chem_eval_line_v2', 1),
        'chem_line_f1': ('chem_eval_line_v2', 2),
    }
    if monitor == 'chem_mean_f1':
        box = stats.get('chem_eval_box')
        line = stats.get('chem_eval_line_v2', stats.get('chem_eval_line'))
        if not isinstance(box, (list, tuple)) or len(box) < 3:
            return None
        if not isinstance(line, (list, tuple)) or len(line) < 3:
            return None
        return (float(box[2]) + float(line[2])) / 2.0

    key_index = metric_indexes.get(monitor)
    if key_index is None:
        return None
    values = stats.get(key_index[0])
    if not isinstance(values, (list, tuple)) or len(values) <= key_index[1]:
        return None
    return float(values[key_index[1]])


class DetSolver(BaseSolver):

    def fit(self, ):
        self.train()
        args = self.cfg

        n_parameters, model_stats = stats(self.cfg)
        print(model_stats)
        print("-"*42 + "Start training" + "-"*43)

        self.self_lr_scheduler = False
        if args.lrsheduler is not None:
            iter_per_epoch = len(self.train_dataloader)
            print("     ## Using Self-defined Scheduler-{} ## ".format(args.lrsheduler))
            self.lr_scheduler = FlatCosineLRScheduler(self.optimizer, args.lr_gamma, iter_per_epoch, total_epochs=args.epoches,
                                                warmup_iter=args.warmup_iter, flat_epochs=args.flat_epoch, no_aug_epochs=args.no_aug_epoch)
            self.self_lr_scheduler = True
        n_parameters = sum([p.numel() for p in self.model.parameters() if p.requires_grad])
        print(f'number of trainable parameters: {n_parameters}')

        top1 = 0
        best_stat = {'epoch': -1, }
        early_cfg = args.early_stopping or {}
        early_enabled = bool(early_cfg.get('enabled', False))
        early_monitor = early_cfg.get('monitor', 'chem_mean_f1')
        early_mode = early_cfg.get('mode', 'max')
        early_patience = int(early_cfg.get('patience', 8))
        early_min_delta = float(early_cfg.get('min_delta', 0.0))
        early_min_epochs = int(early_cfg.get('min_epochs', 0))
        if early_mode not in ('min', 'max'):
            raise ValueError(f"early_stopping.mode must be 'min' or 'max', got {early_mode!r}")
        if early_patience < 1 or early_min_epochs < 0:
            raise ValueError('early_stopping patience must be >= 1 and min_epochs must be >= 0')
        early_best_score = None
        early_bad_epochs = 0
        early_start_epoch = self.last_epoch + 1
        # evaluate again before resume training
        if self.last_epoch > 0:
            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device
            )
            for k, values in test_stats.items():
                if not isinstance(values, (list, tuple)) or len(values) == 0:
                    continue
                best_stat['epoch'] = self.last_epoch
                best_stat[k] = values[0]
                top1 = values[0]
                print(f'best_stat: {best_stat}')

            if early_enabled:
                early_best_score = _early_stopping_score(test_stats, early_monitor)
                if early_best_score is None:
                    raise KeyError(f"early stopping monitor {early_monitor!r} is unavailable in validation stats")
                if self.output_dir:
                    dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best_early_stop.pth')
                print(f'early_stopping baseline: monitor={early_monitor} score={early_best_score:.6f}')

        best_stat_print = best_stat.copy()
        start_time = time.time()
        start_epoch = self.last_epoch + 1
        for epoch in range(start_epoch, args.epoches):

            self.train_dataloader.set_epoch(epoch)
            # self.train_dataloader.dataset.set_epoch(epoch)
            if dist_utils.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)

            if epoch == self.train_dataloader.collate_fn.stop_epoch:
                self.load_resume_state(str(self.output_dir / 'best_stg1.pth'))
                self.ema.decay = self.train_dataloader.collate_fn.ema_restart_decay
                print(f'Refresh EMA at epoch {epoch} with decay {self.ema.decay}')

            train_stats, grad_percentages = train_one_epoch(
                self.self_lr_scheduler,
                self.lr_scheduler,
                self.model,
                self.criterion,
                self.train_dataloader,
                self.optimizer,
                self.device,
                epoch,
                max_norm=args.clip_max_norm,
                print_freq=args.print_freq,
                ema=self.ema,
                scaler=self.scaler,
                lr_warmup_scheduler=self.lr_warmup_scheduler,
                writer=self.writer,
                teacher_model=self.teacher_model, # NEW: Pass teacher model to train_one_epoch
            )

            if not self.self_lr_scheduler:  # update by epoch 
                if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                    self.lr_scheduler.step()

            self.last_epoch += 1
            if dist_utils.is_main_process() and hasattr(self.criterion, 'distill_adaptive_params') and \
                self.criterion.distill_adaptive_params and self.criterion.distill_adaptive_params.get('enabled', False):

                params = self.criterion.distill_adaptive_params
                default_weight = params.get('default_weight')

                avg_percentage = sum(grad_percentages) / len(grad_percentages) if grad_percentages else 0.0

                current_weight = self.criterion.weight_dict.get('loss_distill', 0.0)
                new_weight = current_weight
                reason = 'unchanged'

                if avg_percentage < 1e-6:
                    if default_weight is not None:
                        new_weight = default_weight
                        reason = 'reset_to_default_zero_grad'
                elif epoch >= self.train_dataloader.collate_fn.stop_epoch:
                    if default_weight is not None:
                        new_weight = default_weight
                        reason = 'ema_phase_default'
                else:
                    rho = params['rho']
                    delta = params['delta']
                    lower_bound = rho - delta
                    upper_bound = rho + delta
                    if not (lower_bound <= avg_percentage <= upper_bound):
                        target_percentage = upper_bound if avg_percentage < lower_bound else lower_bound
                        if current_weight > 1e-6:
                            p_current = avg_percentage / 100.0
                            p_target = target_percentage / 100.0
                            numerator = p_target * (1.0 - p_current)
                            denominator = p_current * (1.0 - p_target)
                            if abs(denominator) >= 1e-9:
                                ratio = numerator / denominator
                                ratio = max(ratio, 0.1)  # clamp non-positive to 0.1
                                new_weight = current_weight * ratio
                                new_weight = min(max(new_weight, current_weight / 10.0), current_weight * 10.0)
                                reason = f'adjusted_to_{target_percentage:.2f}%'

                if abs(new_weight - current_weight) > 0:
                    self.criterion.weight_dict['loss_distill'] = new_weight
                print(f"Epoch {epoch}: avg encoder grad {avg_percentage:.2f}% | distill {current_weight:.6f} -> {new_weight:.6f} ({reason})")

            if self.output_dir and epoch < self.train_dataloader.collate_fn.stop_epoch:
                checkpoint_paths = [self.output_dir / 'last.pth']
                # extra checkpoint before LR drop and every 100 epochs
                if (epoch + 1) % args.checkpoint_freq == 0:
                    checkpoint_paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    dist_utils.save_on_master(self.state_dict(), checkpoint_path)

            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device
            )

            should_stop = False
            early_score = None
            if early_enabled:
                early_score = _early_stopping_score(test_stats, early_monitor)
                if early_score is None:
                    raise KeyError(f"early stopping monitor {early_monitor!r} is unavailable in validation stats")
                delta = early_score - early_best_score if early_mode == 'max' else early_best_score - early_score
                if delta > early_min_delta:
                    early_best_score = early_score
                    early_bad_epochs = 0
                    if self.output_dir:
                        dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best_early_stop.pth')
                elif epoch - early_start_epoch + 1 > early_min_epochs:
                    early_bad_epochs += 1
                should_stop = early_bad_epochs >= early_patience
                print(
                    f'early_stopping: monitor={early_monitor} score={early_score:.6f} '
                    f'best={early_best_score:.6f} bad_epochs={early_bad_epochs}/{early_patience}'
                )

            # TODO
            for k, values in test_stats.items():
                is_sequence_metric = isinstance(values, (list, tuple))

                if self.writer and dist_utils.is_main_process() and is_sequence_metric:
                    for i, v in enumerate(values):
                        self.writer.add_scalar(f'Test/{k}_{i}'.format(k), v, epoch)

                if not is_sequence_metric or len(values) == 0:
                    continue

                current_score = values[0]
                if k in best_stat:
                    best_stat['epoch'] = epoch if current_score > best_stat[k] else best_stat['epoch']
                    best_stat[k] = max(best_stat[k], current_score)
                else:
                    best_stat['epoch'] = epoch
                    best_stat[k] = current_score

                if best_stat[k] > top1:
                    best_stat_print['epoch'] = epoch
                    top1 = best_stat[k]
                    if self.output_dir:
                        if epoch >= self.train_dataloader.collate_fn.stop_epoch:
                            dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best_stg2.pth')
                        else:
                            dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best_stg1.pth')

                best_stat_print[k] = max(best_stat[k], top1)
                print(f'best_stat: {best_stat_print}')  # global best

                if best_stat['epoch'] == epoch and self.output_dir:
                    if epoch >= self.train_dataloader.collate_fn.stop_epoch:
                        if current_score > top1:
                            top1 = current_score
                            dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best_stg2.pth')
                    else:
                        top1 = max(current_score, top1)
                        dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best_stg1.pth')

                elif epoch >= self.train_dataloader.collate_fn.stop_epoch:
                    best_stat = {'epoch': -1, }
                    self.ema.decay -= 0.0001
                    self.load_resume_state(str(self.output_dir / 'best_stg1.pth'))
                    print(f'Refresh EMA at epoch {epoch} with decay {self.ema.decay}')


            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'test_{k}': v for k, v in test_stats.items()},
                'epoch': epoch,
                'n_parameters': n_parameters
            }
            if early_enabled:
                log_stats.update({
                    'early_stopping_monitor': early_monitor,
                    'early_stopping_score': early_score,
                    'early_stopping_best': early_best_score,
                    'early_stopping_bad_epochs': early_bad_epochs,
                })

            if self.output_dir and dist_utils.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:
                    (self.output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                    self.output_dir / "eval" / name)

                if early_enabled:
                    early_state = {
                        'monitor': early_monitor,
                        'mode': early_mode,
                        'best_score': early_best_score,
                        'bad_epochs': early_bad_epochs,
                        'patience': early_patience,
                        'min_delta': early_min_delta,
                        'epoch': epoch,
                        'stopped': should_stop,
                    }
                    (self.output_dir / 'early_stopping.json').write_text(
                        json.dumps(early_state, indent=2) + '\n'
                    )

            if should_stop:
                print(f'Early stopping at epoch {epoch}; best {early_monitor}={early_best_score:.6f}')
                break

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))


    def val(self, ):
        self.eval()

        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(module, self.criterion, self.postprocessor,
                self.val_dataloader, self.evaluator, self.device)

        if self.output_dir and coco_evaluator is not None and hasattr(coco_evaluator, 'coco_eval') and "bbox" in coco_evaluator.coco_eval:
            dist_utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")

        return test_stats, coco_evaluator


    def state_dict(self):
        """State dict, train/eval"""
        state = {}
        state['date'] = datetime.datetime.now().isoformat()

        # For resume
        state['last_epoch'] = self.last_epoch

        for k, v in self.__dict__.items():
            if k == 'teacher_model':
                continue
            if hasattr(v, 'state_dict'):
                v = dist_utils.de_parallel(v)
                state[k] = v.state_dict()

        return state
