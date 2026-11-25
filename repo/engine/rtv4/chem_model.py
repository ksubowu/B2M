import torch.nn as nn

from ..core import register


def _remap_line_outputs(outputs):
    remapped = {
        'pred_line_logits': outputs['pred_logits'],
        'pred_lines': outputs['pred_boxes'],
    }

    if 'aux_outputs' in outputs:
        remapped['line_aux_outputs'] = [
            {
                'pred_line_logits': item['pred_logits'],
                'pred_lines': item['pred_boxes'],
            }
            for item in outputs['aux_outputs']
        ]

    if 'enc_aux_outputs' in outputs:
        remapped['line_enc_aux_outputs'] = [
            {
                'pred_line_logits': item['pred_logits'],
                'pred_lines': item['pred_boxes'],
            }
            for item in outputs['enc_aux_outputs']
        ]

    return remapped


@register()
class ChemRTv4(nn.Module):
    __inject__ = ['backbone', 'encoder', 'box_decoder', 'line_decoder']

    def __init__(
        self,
        backbone: nn.Module,
        encoder: nn.Module,
        box_decoder: nn.Module,
        line_decoder: nn.Module,
    ):
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.box_decoder = box_decoder
        self.line_decoder = line_decoder

    def _build_line_targets(self, targets):
        if targets is None:
            return None
        line_targets = []
        for target in targets:
            line_targets.append({
                'labels': target['line_labels'],
                'boxes': target['lines'],
            })
        return line_targets

    def forward(self, x, targets=None, teacher_encoder_output=None):
        x_backbone = self.backbone(x)
        encoder_output = self.encoder(x_backbone)

        student_distill_output = None
        if self.training and isinstance(encoder_output, tuple) and len(encoder_output) == 2:
            x_fpn_features, student_distill_output = encoder_output
        else:
            x_fpn_features = encoder_output

        box_outputs = self.box_decoder(x_fpn_features, targets=targets)
        line_outputs = self.line_decoder(x_fpn_features, targets=self._build_line_targets(targets))

        outputs = dict(box_outputs)
        outputs.update(_remap_line_outputs(line_outputs))

        if self.training and student_distill_output is not None and teacher_encoder_output is not None:
            outputs['student_distill_output'] = student_distill_output
            outputs['teacher_encoder_output'] = teacher_encoder_output

        return outputs

    def deploy(self):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self
