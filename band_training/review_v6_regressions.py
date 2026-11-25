"""Make aligned original / GT / predicted panels for new replay regressions."""
import argparse,gzip,json,zipfile,html
from pathlib import Path
from collections import Counter
from PIL import Image,ImageDraw,ImageFont
import numpy as np
from .mixed_data import letterbox_transform,ATOM_CLASSES


def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    with gzip.open(a.baseline,'rt') as f:before={d['sample_id']:d['components'] for d in map(json.loads,f)}
    pools={'rare_atom_false_positive':[],'wrong_aromatic_bond':[],'other_atom_missed':[]};totals=Counter()
    with gzip.open(a.candidate,'rt') as f:
        for d in map(json.loads,f):
            sid=d['sample_id'];mapping={int(k):v for k,v in d['atom_mapping'].items()};gtpairs={tuple(sorted((b['begin'],b['end']))):b['category_id']-1 for b in d['ann']['bonds']}
            rare=[dict(label=x['label'],name=ATOM_CLASSES[x['label']],score=x['score'],bbox=x['bbox']) for i,x in enumerate(d['atoms']) if i not in mapping and x['label'] in [9,10,11,12]]
            aromatic=[]
            for e in d['bonds']:
                pair=tuple(sorted((mapping.get(e['begin'],-1),mapping.get(e['end'],-1))))
                if e['label']==3 and pair in gtpairs and gtpairs[pair]!=3:aromatic.append(dict(predicted=3,truth=gtpairs[pair],instance=e['bond_instance'],score=e['score']))
            missing=[x for i,x in enumerate(d['ann']['atoms']) if i not in mapping.values() and x['category_id']==0]
            regressed=bool(before[sid]['legacy_full_exact'] and not d['components']['legacy_full_exact'])
            for key,items in [('rare_atom_false_positive',rare),('wrong_aromatic_bond',aromatic),('other_atom_missed',missing)]:
                totals[key+'_instances']+=len(items);totals[key+'_images']+=bool(items)
                if items:pools[key].append((regressed,max([x.get('score',0.) for x in items]),d,items))
    selected=[];seen=set()
    for key,pool in pools.items():
        taken=0
        for regressed,score,d,items in sorted(pool,key=lambda x:(x[0],x[1]),reverse=True):
            if d['sample_id'] in seen:continue
            selected.append((key,d,items));seen.add(d['sample_id']);taken+=1
            if taken>=5:break
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15);cards=[]
    for i,(kind,d,items) in enumerate(selected):
        raw=json.loads((a.dataset/d['record']['annotation']).read_text());band=json.loads((a.dataset/d['record']['band_target']).read_text())
        with Image.open(a.dataset/d['record']['image']) as im:
            im=im.convert('RGB');scale,offset,shape=letterbox_transform(*im.size);canvas=Image.new('RGB',(640,640),'white');canvas.paste(im.resize(shape,Image.Resampling.LANCZOS),tuple(offset))
        panels=[canvas,canvas.copy(),canvas.copy()]
        for j,objects in [(1,[dict(label=x['category_id'],bbox=x['bbox']) for x in d['ann']['atoms']]),(2,d['atoms'])]:
            draw=ImageDraw.Draw(panels[j]);matched={int(k) for k in d['atom_mapping']}
            for idx,obj in enumerate(objects):
                x,y,w,h=obj['bbox'];color='blue' if j==1 or idx in matched else 'red';draw.rectangle((x,y,x+w,y+h),outline=color,width=1)
                if obj['label']!=1:draw.text((x,y-12),ATOM_CLASSES[obj['label']],fill=color)
        quads=(np.asarray(band['quads']).reshape(-1,4,2)*scale+offset).tolist();labels=band['labels']
        for j,quads,labels in [(1,quads,labels),(2,[x['quad'] for x in d['instances']],[x['label'] for x in d['instances']])]:
            draw=ImageDraw.Draw(panels[j]);names=['S','D','T','A','W','dash']
            for q,label in zip(quads,labels):
                points=[tuple(x) for x in q];draw.line(points+[points[0]],fill='green' if j==1 else 'orange',width=1);center=np.mean(q,axis=0);draw.text(tuple(center),names[label],fill='green' if j==1 else 'darkorange')
        result=Image.new('RGB',(1920,710),'white');draw=ImageDraw.Draw(result)
        for j,title in enumerate(['Original (same 640 letterbox)','Ground truth boxes + bond bands','Replay B epoch3 prediction (red = unmatched atom)']):result.paste(panels[j],(640*j,30));draw.text((640*j+5,7),title,font=font,fill='black')
        draw.text((5,676),kind+' | '+d['sample_id'],font=font,fill='black')
        name=f'{i+1:02d}_{kind}.png';result.save(a.output/name);cards.append(dict(image=name,kind=kind,sample_id=d['sample_id'],details=items,baseline_components=before[d['sample_id']],candidate_components=d['components']))
    (a.output/'samples.json').write_text(json.dumps(cards,ensure_ascii=False,indent=2));(a.output/'counts.json').write_text(json.dumps(dict(totals),indent=2))
    page='<html><head><meta charset="utf-8"></head><body><h2>新增错误检查</h2><p>原图 | GT box+band | B组预测；红框表示未匹配的预测原子，蓝框为匹配原子。键型 S/D/T/A/W/dash。未展示的电荷不用于本页诊断。</p>'
    for card in cards:page+='<p>'+html.escape(card['kind']+' '+card['sample_id'])+'</p><img style="width:100%" src="'+card['image']+'">'
    (a.output/'index.html').write_text(page+'</body></html>')
    with zipfile.ZipFile(a.output.parent/'error_review.zip','w',zipfile.ZIP_DEFLATED) as z:
        for path in a.output.iterdir():z.write(path,path.relative_to(a.output.parent))
    print(json.dumps(dict(samples=len(cards),counts=dict(totals))))


if __name__=='__main__':main()
