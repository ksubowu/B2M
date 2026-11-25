"""Background supervisor: bounded parallel controls, results, then epoch40 audit."""
import argparse,json,os,subprocess,sys,time,traceback
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root.resolve();project=Path.cwd();py=sys.executable
    smoke=json.loads((r/'smoke_heads/results.json').read_text())
    assert smoke['status']=='completed'
    assert not smoke['changed_modules']['backbone'] and not smoke['changed_modules']['encoder']
    assert all(smoke['changed_modules'][k] for k in ['boxes','line_head','band_head'])
    assert all(smoke['gradients_seen'].get(k,0)>0 for k in ['boxes','shared_bond','line_head','band_head'])
    children=[];logs=[]
    specs=[('uniform_heads','0','0,2,4,6,8,10,12,14,16,18,20,22'),('replay_heads','5','160,162,164,166,168,170,172,174,176,178,180,182')]
    for arm,gpu,cores in specs:
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
        command=['taskset','-c',cores,'nice','-n','5',py,'-u','-m','band_training.train_v6_continue',
            '--dataset','/mnt/hdd2/bo/work/cvs/chemcv_data/real/indigo_diverse_v6_20261121/dataset',
            '--preflight',str(project/'runs/v6_continue_20261121/preflight'),'--init',str(project/'runs/line_neighborhood_formal_v2_20261119/best_joint.pt'),
            '--output',str(r/arm),'--training-manifest',str(r/(arm+'_train.json')),'--epochs','3','--batch-size','16','--workers','4',
            '--freeze-feature-steps','10000000','--box-lr','0.000002','--shared-bond-lr','0.000003','--line-lr','0.000005','--band-lr','0.000005',
            '--warmup-steps','200','--patience','0','--checkpoint-every','1000','--memory-fraction','.3','--seed','20261124']
        log=(r/(arm+'.log')).open('w');logs.append(log);proc=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT);children.append((arm,proc))
        print('STARTED',arm,'pid',proc.pid,'gpu',gpu,flush=True)
    while any(p.poll() is None for _,p in children):
        subprocess.run([py,'-m','band_training.summarize_v6_recovery','--root',str(r)],check=True)
        time.sleep(30)
    codes={arm:p.wait() for arm,p in children};(r/'exit_codes.json').write_text(json.dumps(codes,indent=2))
    for log in logs:log.close()
    subprocess.run([py,'-m','band_training.summarize_v6_recovery','--root',str(r)],check=True)
    # Existing 40-epoch run is independent and is allowed to finish unchanged.
    final=project/'runs/v6_18epochs_20261124/validation/epoch_40/prediction_cache/predictions.jsonl.gz'
    deadline=time.time()+7200
    while not final.exists() and time.time()<deadline:time.sleep(30)
    if final.exists():
        subprocess.run([py,'-m','band_training.diagnose_v6_recovery','--baseline',str(project/'runs/v6_continue_20261121/formal/validation/baseline/prediction_cache/predictions.jsonl.gz'),'--candidate',str(final),'--output',str(r/'diagnosis_epoch40')],check=True)
    subprocess.run([py,'-m','band_training.summarize_v6_recovery','--root',str(r)],check=True)
    if any(codes.values()):raise RuntimeError('Experiment failed: '+str(codes))
    print('COMPLETED',codes,flush=True)


if __name__=='__main__':main()
