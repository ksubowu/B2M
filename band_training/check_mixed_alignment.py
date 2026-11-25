"""Audit actual loader targets after resize/letterbox, before any optimizer step."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
from .mixed_data import MixedIndigoDataset,letterbox_transform,prepare_annotation
from .model import targets_for


def verify_target(ann,source_target):
    metric,box_target,labels,quads=prepare_annotation(ann,source_target)
    scale,offset,_=letterbox_transform(ann['width'],ann['height'])
    source_boxes=np.asarray([a['bbox'] for a in ann['atoms']]+[q['bbox'] for q in ann['charges']],float)
    expected=source_boxes.reshape(-1,2,2)*scale+offset
    actual=box_target['boxes'].numpy()*640
    xyxy=np.stack((actual[:,:2]-actual[:,2:]/2,actual[:,:2]+actual[:,2:]/2),1)
    box_error=float(np.abs(xyxy-expected).max(initial=0))
    band_target=targets_for(labels,quads,'cpu',resolve_collisions=True)
    reconstructed=[];shifted=0
    for label,yi,xi,off,geom in band_target[1]:
        center=(torch.tensor([xi,yi])+off)*4
        reconstructed.append((geom.reshape(4,2)*32+center).numpy())
        natural=np.floor(np.asarray(quads[len(reconstructed)-1]).mean(0)/4).astype(int)
        shifted+=int(tuple(natural)!=(xi,yi))
    assert len(reconstructed)==len(quads)
    band_error=float(np.abs(np.asarray(reconstructed)-np.asarray(quads)).max(initial=0))
    assert box_error<.001,('Box resize misalignment',box_error)
    assert band_error<.001,('Band resize misalignment',band_error)
    assert np.isfinite(np.asarray(quads)).all()
    return dict(box_target_max_error_px=box_error,band_target_max_error_px=band_error,shifted_heatmap_anchors=shifted)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    candidates=[];stats=dict(samples=0,atoms=0,charges=0,bonds=0,shifted_heatmap_anchors=0,
                            max_box_target_error_px=0.,max_band_target_error_px=0.)
    largest=(-1,None);charged=None;color=None;stereo=None;variant=None;seen={}
    datasets={s:MixedIndigoDataset(args.dataset,s,augment=False) for s in ['train','val']}
    for split in ['train','val']:
        for index,row in enumerate(datasets[split].records):
            asset=(row['annotation'],row['band_target'])
            if asset not in seen:
                ann=json.loads((args.dataset/row['annotation']).read_text());target=json.loads((args.dataset/row['band_target']).read_text())
                result=verify_target(ann,target);seen[asset]=(ann,target,result)
            else:ann,target,result=seen[asset]
            stats['samples']+=1;stats['atoms']+=len(ann['atoms']);stats['charges']+=len(ann['charges']);stats['bonds']+=len(ann['bonds'])
            stats['shifted_heatmap_anchors']+=result['shifted_heatmap_anchors']
            stats['max_box_target_error_px']=max(stats['max_box_target_error_px'],result['box_target_max_error_px'])
            stats['max_band_target_error_px']=max(stats['max_band_target_error_px'],result['band_target_max_error_px'])
            key=(split,index)
            if index==0:candidates.append(key)
            if len(ann['atoms'])>largest[0]:largest=(len(ann['atoms']),key)
            if ann['charges'] and charged is None:charged=key
            if row['is_color'] and row['dummy_applied'] and color is None:color=key
            if any(label in [4,5] for label in target['labels']) and stereo is None:stereo=key
            if 'charge_augmentation' in row['image'] and variant is None:variant=key
            if stats['samples']%1000==0:print('ALIGNMENT',stats['samples'],flush=True)
    candidates=list(dict.fromkeys(candidates+[largest[1],charged,color,stereo,variant]));candidates=[k for k in candidates if k]
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16);samples=[]
    for number,(split,index) in enumerate(candidates):
        image,meta=datasets[split][index]
        rgb=(image.permute(1,2,0).numpy()*255).round().astype(np.uint8);base=Image.fromarray(rgb)
        boxes=base.copy();draw=ImageDraw.Draw(boxes)
        for a in meta['ann']['atoms']:
            x,y,w,h=a['bbox'];draw.rectangle((x,y,x+w,y+h),outline='#0076c0',width=2)
        for q in meta['ann']['charges']:
            x,y,w,h=q['bbox'];draw.rectangle((x,y,x+w,y+h),outline='#ed1059',width=2)
        bands=base.convert('RGBA');layer=Image.new('RGBA',base.size);draw=ImageDraw.Draw(layer)
        palette=[(210,55,55),(25,150,80),(70,90,210),(185,55,180),(220,130,0),(0,155,175)]
        for label,quad in zip(meta['labels'],meta['quads']):
            poly=[tuple(p) for p in quad];c=palette[label]
            draw.polygon(poly,fill=c+(30,),outline=c+(220,),width=1)
        bands=Image.alpha_composite(bands,layer).convert('RGB')
        canvas=Image.new('RGB',(1920,720),'white');draw=ImageDraw.Draw(canvas)
        for j,(im,title) in enumerate(zip([base,boxes,bands],['Resized image (RGB + letterbox)','Same transform: atom / charge boxes','Same transform: band regions'])):
            canvas.paste(im,(j*640,40));draw.text((j*640+8,8),title,font=font,fill='black')
        draw.text((8,687),meta['record']['sample_id']+' | '+str(len(meta['ann']['atoms']))+' atoms / '+str(len(meta['labels']))+' bands',font=font,fill='black')
        path=args.output/f'alignment_{number:02d}.png';canvas.save(path)
        row=meta['record'];source=json.loads((args.dataset/row['annotation']).read_text());scale,offset,shape=letterbox_transform(source['width'],source['height'])
        samples.append(dict(sample_id=row['sample_id'],split=split,image_size_before=[source['width'],source['height']],
                            resize_shape=list(shape),scale_xy=scale.tolist(),padding_xy=offset.tolist(),
                            color=row['is_color'],preview=str(path),rgb_channels_differ=bool((rgb[:,:,0]!=rgb[:,:,1]).any())))
    report=dict(status='passed',**stats,unique_assets=len(seen),
                manifest_sha256={s:hashlib.sha256(d.manifest.read_bytes()).hexdigest() for s,d in datasets.items()},previews=samples,
                policy='All geometric targets follow the same actual rounded RGB resize and letterbox translation. No molecule rotation/reflection. Displaced heatmap anchors recover unchanged true regions through offset targets.')
    (args.output/'alignment_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
