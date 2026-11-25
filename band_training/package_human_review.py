"""Package unaugmented human Fix originals, exact boxes and image-derived band drafts.

Human Fix records contain bond centers/ends and classes, not human-confirmed
band masks. Derived masks are explicitly drafts and must not be used as reviewed
training truth. The source images/annotations are copied byte for byte.
"""
import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont


def draft_bands(image,annotation):
    gray=np.asarray(image.convert('L'))
    _,ink=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    ink=ink>0;h,w=ink.shape
    glyphs=np.zeros((h,w),bool)
    def exclude(box,xywh=False):
        x0,y0,x1,y1=box
        if xywh:x1+=x0;y1+=y0
        if x1<=x0 or y1<=y0:return
        glyphs[max(0,int(y0)-1):min(h,int(np.ceil(y1))+2),max(0,int(x0)-1):min(w,int(np.ceil(x1))+2)]=True
    for atom in annotation['atoms']:
        if atom.get('symbol')!='C':exclude(atom['bbox'],True)
    for q in annotation.get('charges',[]):exclude(q['bbox'])
    for text in annotation.get('ocr',{}).get('items',[]):
        box=text.get('bbox')
        if box and len(box)==4 and max(box)>0:exclude(box)
    usable=ink&~glyphs
    score=np.full((h,w),np.inf,np.float32)
    second=np.full((h,w),np.inf,np.float32)
    owners=np.zeros((h,w),np.uint16)
    for i,bond in enumerate(annotation['bonds']):
        a=np.asarray(bond['line'][:2],float);b=np.asarray(bond['line'][2:],float)
        delta=b-a;length=float(np.linalg.norm(delta))
        if length<1:continue
        axis=delta/length;normal=np.array([-axis[1],axis[0]])
        factor={1:.09,2:.18,3:.22,4:.24,5:.22,6:.22}.get(bond['category_id'],.18)
        halfwidth=max(2.,min(18.,length*factor))
        low=np.floor(np.minimum(a,b)-halfwidth-1).astype(int)
        high=np.ceil(np.maximum(a,b)+halfwidth+2).astype(int)
        x0,y0=max(0,low[0]),max(0,low[1]);x1,y1=min(w,high[0]),min(h,high[1])
        if x1<=x0 or y1<=y0:continue
        yy,xx=np.mgrid[y0:y1,x0:x1];dx=xx-a[0];dy=yy-a[1]
        along=dx*axis[0]+dy*axis[1];perp=np.abs(dx*normal[0]+dy*normal[1])
        valid=usable[y0:y1,x0:x1]&(along>=0)&(along<=length)&(perp<=halfwidth)
        cost=perp/halfwidth+.04*((along-length/2)/length)**2
        old=score[y0:y1,x0:x1];old2=second[y0:y1,x0:x1]
        better=valid&(cost<old)
        old2[:]=np.where(better,old,np.where(valid,np.minimum(old2,cost),old2))
        old[:]=np.where(better,cost,old)
        owner=owners[y0:y1,x0:x1];owner[better]=i+1
    ambiguous=np.zeros((h,w),bool)
    both=np.isfinite(score)&np.isfinite(second)
    ambiguous[both]=(second[both]-score[both])<.15
    bands=[]
    for i,b in enumerate(annotation['bonds']):
        ys,xs=np.nonzero(owners==i+1)
        poly=[]
        if len(xs)>=3:
            poly=cv2.convexHull(np.column_stack((xs,ys)).astype(np.int32)).reshape(-1,2).tolist()
        bands.append(dict(index=i,category_id=b['category_id'],source_label=b.get('source_label'),
                          human_line=b['line'],human_begin=b['begin'],human_end=b['end'],
                          ink_pixels=len(xs),region_polygon=poly,
                          ambiguous_pixels=int(ambiguous[owners==i+1].sum()),
                          status='DRAFT_NOT_HUMAN_VERIFIED' if len(xs) else 'NO_INK_REQUIRES_REVIEW'))
    return owners,ambiguous,bands


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    for name in ['originals','human_annotations','boxes','bands_draft','comparisons','derived_annotations','masks']:
        (args.output/name).mkdir()
    manifest=json.loads((args.dataset/'manifest.json').read_text())
    native=[r for r in manifest if r['source']=='webapp_human_fix' and r.get('augmentation',{}).get('kind')=='native']
    assert len(native)==len(set(r['group'] for r in manifest if r['source']=='webapp_human_fix'))
    fontpath='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    overview=[];records=[]
    for row in native:
        name=Path(row['file_name']).stem
        original_path=args.dataset/row['image'];truth_path=args.dataset/row['truth']
        ann=json.loads(truth_path.read_text());image=Image.open(original_path).convert('RGB');w,h=image.size
        assert image.size==tuple(ann['image_size'])
        shutil.copy2(original_path,args.output/'originals'/f'{name}.png')
        shutil.copy2(truth_path,args.output/'human_annotations'/f'{name}.json')
        owner,ambiguous,bands=draft_bands(image,ann)
        np.savez_compressed(args.output/'masks'/f'{name}.npz',bond_owner=owner,ambiguous=ambiguous)
        derived=dict(status='DRAFT_NOT_HUMAN_VERIFIED',image_size=[w,h],
                     source='Original image ink constrained by human-confirmed bond endpoints/classes; text/charge boxes excluded.',
                     limitations=['No original human band masks exist. These are proposals for visual checking.',
                                  'Nearby/crossing bonds, aromatic circles and text can lead to incorrect pixel assignment.',
                                  'Do not use as human-reviewed band training truth.'],
                     owner_encoding='0=background/unassigned; n>0=source bond index n-1',bonds=bands)
        (args.output/'derived_annotations'/f'{name}.json').write_text(json.dumps(derived,indent=2))
        boxes=image.copy();draw=ImageDraw.Draw(boxes)
        for a in ann['atoms']:
            x,y,bw,bh=a['bbox'];draw.rectangle((x,y,x+bw,y+bh),outline='#087fc5',width=2)
        for q in ann.get('charges',[]):draw.rectangle(q['bbox'],outline='#e10061',width=2)
        boxes.save(args.output/'boxes'/f'{name}.png')
        rgb=np.asarray(image).astype(float).copy();colors=[]
        for i,b in enumerate(bands):
            color=cv2.cvtColor(np.uint8([[[int(i*137.5)%180,210,220]]]),cv2.COLOR_HSV2RGB)[0,0]
            colors.append(tuple(map(int,color)))
            ink=owner==i+1
            rgb[ink]=rgb[ink]*.28+color*.72
        overlay=Image.fromarray(rgb.clip(0,255).astype(np.uint8));draw=ImageDraw.Draw(overlay)
        for b,color in zip(bands,colors):
            poly=[tuple(v) for v in b['region_polygon']]
            if len(poly)>1:draw.line(poly+[poly[0]],fill=color,width=1)
        overlay.save(args.output/'bands_draft'/f'{name}.png')
        # All three images have identical coordinates and dimensions; no scaling.
        fontsize=max(12,min(22,w//26));font=ImageFont.truetype(fontpath,fontsize)
        small=ImageFont.truetype(fontpath,max(10,min(16,w//33)))
        header=max(44,fontsize*2+8);footer=max(68,fontsize*3)
        compare=Image.new('RGB',(3*w,h+header+footer),'white');draw=ImageDraw.Draw(compare)
        for i,(panel,title) in enumerate(zip([image,boxes,overlay],['Original (native)','Original + HUMAN boxes','Original + band DRAFT'])):
            compare.paste(panel,(i*w,header));draw.text((i*w+8,8),title,font=font,fill='#19263a')
        draw.text((8,header+h+8),f'{name} | {len(ann["atoms"])} atoms, {len(ann["charges"])} charges, {len(bands)} bonds',font=small,fill='#19263a')
        draw.text((8,header+h+30),'Blue: human atom boxes; pink: human charge boxes. Bands: automatically extracted proposals, NOT human-confirmed masks.',font=small,fill='#a02b22')
        compare.save(args.output/'comparisons'/f'{name}_triptych.png')
        # Verify original panel byte values and unchanged source artifacts.
        assert np.array_equal(np.asarray(compare.crop((0,header,w,header+h))),np.asarray(image))
        assert hashlib.sha256(original_path.read_bytes()).digest()==hashlib.sha256((args.output/'originals'/f'{name}.png').read_bytes()).digest()
        scaled=compare.copy();scaled.thumbnail((1800,900));overview.append(scaled)
        record=dict(file_name=name,atoms=len(ann['atoms']),charges=len(ann['charges']),bonds=len(bands),
                    annotation_version_status='TRANSFER_TRAINING_SNAPSHOT_LATEST_UNVERIFIED',
                    source_revision=ann.get('revision'),source_saved_at_utc=ann.get('saved_at_utc'),
                    image_size=[w,h],augmentation=row['augmentation'],source_manifest=row,
                    original=f'originals/{name}.png',human_annotation=f'human_annotations/{name}.json',
                    comparison=f'comparisons/{name}_triptych.png',
                    band_annotation_status='DRAFT_NOT_HUMAN_VERIFIED',
                    empty_draft_bands=[b['index'] for b in bands if not b['ink_pixels']])
        records.append(record);print(json.dumps({k:record[k] for k in ['file_name','atoms','charges','bonds','empty_draft_bands']}),flush=True)
    sheet=Image.new('RGB',(max(im.width for im in overview),sum(im.height for im in overview)+16*(len(overview)-1)),'#dce2e9')
    y=0
    for im in overview:sheet.paste(im,(0,y));y+=im.height+16
    sheet.save(args.output/'overview.jpg',quality=94)
    (args.output/'manifest.json').write_text(json.dumps(records,indent=2,ensure_ascii=False))
    note='''人工确认原始样本检查包\n\n共 5 个 WebApp human Fix 的 native 原始样本，不包含旋转、缩放、加噪等增强副本。\noriginals/ 与 human_annotations/ 均从转移包逐字节复制，未重绘、未修改。\ncomparisons/ 中每个 PNG 都是单张横向拼图：原图 | 原图+人工 boxes | 原图+band 草稿。\n三个面板在原始分辨率下逐像素对齐；overview.jpg 仅是缩略总览。\n蓝框=人工原子框，粉框=人工电荷框。原子 bbox 为 xywh，电荷 bbox 为 xyxy。\n\n重要：源数据只有人工确认的原子/电荷框、化学键类别、连接原子与中心线端点，没有人工确认的 band mask。\n第三栏是基于原图笔画与人工键端点约束自动提取的 band 草稿，不是模型预测，也不是已确认的像素真值。\n彩色笔画及其凸包仅供复核；相邻键、交叉、芳香圆环、文字附近可能分配不准。\nderived_annotations/ 与 masks/ 明确标为 DRAFT_NOT_HUMAN_VERIFIED，不能直接作为人工确认的 band 训练标注。\nmasks/*.npz: bond_owner=0 表示背景/未分配，n>0 表示原 JSON 中第 n-1 条键；ambiguous 表示归属近似打平的像素。\n\n其中 HF_75b7bddb178b_native 有 156 个原子和 164 条键，适合检查复杂结构。\n'''
    (args.output/'README.txt').write_text(note,encoding='utf-8')
    archive=args.output.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for file in sorted(args.output.rglob('*')):
            if file.is_file():z.write(file,Path(args.output.name)/file.relative_to(args.output))
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    print('ARCHIVE',archive,'BYTES',archive.stat().st_size,flush=True)

if __name__=='__main__':main()
