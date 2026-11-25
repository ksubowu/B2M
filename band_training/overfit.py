"""Training-only representative overfit check; does not alter the main model."""
import argparse
import json
from pathlib import Path
import torch
from .train import to_device,train_band,predictions,evaluate,write
from .model import BandDecoder


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args();run=args.run
    torch.set_num_threads(2);torch.manual_seed(20261115)
    records=to_device(torch.load(run/'feature_cache.pt',map_location='cpu',weights_only=False),'cuda')
    train=[i for i,r in enumerate(records) if r['row']['split']=='train'];selected=[]
    for label in [4,5,2,1,0]:
        candidate=next((i for i in train if i not in selected and label in records[i]['band']['labels']),None)
        if candidate is not None:selected.append(candidate)
    for i in train:
        if len(selected)==8:break
        if i not in selected:selected.append(i)
    model=BandDecoder().cuda()
    with (run/'tiny_stereo_history.jsonl').open('w') as log:
        train_band(model,records,selected,300,8,1e-3,log,'tiny_all_styles',20261115)
    metrics=evaluate(records,selected,predictions(model,records,selected),.25)
    metrics['training_image_rows']=[records[i]['row']['source_row'] for i in selected]
    metrics['purpose']='Training-only overfit diagnostic including wedge/dash/triple; does not alter the main checkpoint.'
    write(run/'tiny_all_styles.json',metrics)
    print(json.dumps(metrics,indent=2))

if __name__=='__main__':main()
