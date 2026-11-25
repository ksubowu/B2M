"""Package native-resolution GT overlays for the unique failed training assets."""
import argparse,csv,hashlib,html,io,json,re,shutil,zipfile
from collections import OrderedDict
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from .mixed_data import letterbox_transform


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--failures',type=Path,required=True);p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--overwrite',action='store_true');args=p.parse_args()
    failures=[f for f in json.loads(args.failures.read_text()) if f['split']=='train']
    grouped=OrderedDict()
    for failure in failures:grouped.setdefault(failure['record']['annotation'],[]).append(failure)
    assert len(grouped)==56 and len(failures)==127
    args.output.mkdir(parents=True,exist_ok=args.overwrite)
    for name in ['comparisons','annotations','band_targets']:(args.output/name).mkdir(exist_ok=args.overwrite)
    fontfile='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    palette=['#208845','#2872ce','#963fb4','#009998','#d78600','#874244']
    manifest=[];thumbnails=[]
    for number,(annotation_file,cases) in enumerate(grouped.items(),1):
        record=cases[0]['record'];imagepath=(args.dataset/record['image']).resolve()
        annpath=(args.dataset/annotation_file).resolve();targetpath=(args.dataset/record['band_target']).resolve()
        imagebytes=imagepath.read_bytes();annbytes=annpath.read_bytes();targetbytes=targetpath.read_bytes()
        annotation=json.loads(annbytes);target=json.loads(targetbytes)
        original=Image.open(io.BytesIO(imagebytes)).convert('RGB');w,h=original.size
        assert (w,h)==(annotation['width'],annotation['height'])
        quads=np.asarray(target['quads'],float);labels=target['labels']
        assert len(quads)==len(annotation['bonds'])==len(annotation['bands'])
        assert np.allclose(quads,np.asarray([b['quad'] for b in annotation['bands']]),atol=.001)
        scale,offset,_=letterbox_transform(w,h)
        audit_error=max(float(np.max(np.abs(quads[f['band_index']]*scale+offset-np.asarray(f['quad'])))) for f in cases)
        assert audit_error<.001,'Source annotations changed since assembly audit'
        line=max(1,round(min(w,h)/600));fontsize=max(11,min(23,round(min(w,h)/45)))
        font=ImageFont.truetype(fontfile,fontsize);small=ImageFont.truetype(fontfile,max(10,fontsize-2))
        header=fontsize*3+20;footer=fontsize*4+20
        involved={idx for c in cases for idx in c['expected']+(c['predicted'] or [])}
        boxes=original.copy();bd=ImageDraw.Draw(boxes)
        for atom_index,atom in enumerate(annotation['atoms']):
            x1,y1,x2,y2=atom['bbox'];bd.rectangle([x1,y1,x2,y2],outline='#0076c0',width=line)
            if atom_index in involved:
                bd.text((max(0,x1),max(0,y1-fontsize)),str(atom['id']),font=small,fill='#0076c0',stroke_width=1,stroke_fill='white')
        for charge in annotation['charges']:
            x1,y1,x2,y2=charge['bbox'];bd.rectangle([x1,y1,x2,y2],outline='#e10077',width=line)
            bd.text((max(0,x1),max(0,y1-fontsize)),f"{int(charge['value']):+d}",font=small,fill='#e10077',stroke_width=1,stroke_fill='white')
        overlay=Image.new('RGBA',original.size);d=ImageDraw.Draw(overlay)
        bad={c['band_index'] for c in cases}
        for bi,(label,quad) in enumerate(zip(labels,quads)):
            color=palette[label];rgb=tuple(int(color[i:i+2],16) for i in [1,3,5]);poly=[tuple(v) for v in quad]
            d.polygon(poly,fill=rgb+(25,),outline=rgb+(220,),width=line)
        for bi in sorted(bad):
            poly=[tuple(v) for v in quads[bi]];d.line(poly+[poly[0]],fill=(255,0,0,255),width=line+1)
            cx,cy=quads[bi].mean(0);d.text((max(0,cx),max(0,cy-fontsize)),f'B{bi}',font=small,fill='red',stroke_width=1,stroke_fill='white')
        bands=Image.alpha_composite(original.convert('RGBA'),overlay).convert('RGB')
        canvas=Image.new('RGB',(3*w,header+h+footer),'white');d=ImageDraw.Draw(canvas)
        titles=['Original training image','GT atom / charge boxes','GT bond bands']
        subtitles=['Native pixels (no resizing)','Blue: atoms; IDs near failures; pink: charge','Red outline / B#: assembly mismatch']
        for col,(panel,title,subtitle) in enumerate(zip([original,boxes,bands],titles,subtitles)):
            canvas.paste(panel,(col*w,header));d.text((col*w+6,5),title,font=font,fill='black')
            d.text((col*w+6,fontsize+12),subtitle,font=small,fill='#555555')
        asset=record.get('asset_sample_id') or record['sample_id']
        d.text((8,header+h+6),f'{number:02d}/56 | {asset}',font=font,fill='black')
        failed_ids=', '.join(str(i) for i in sorted(bad))
        d.text((8,header+h+fontsize+14),f'Assembly mismatch band indices (0-based): {failed_ids}',font=small,fill='#b00000')
        d.text((8,header+h+fontsize*2+20),'All overlays are ground-truth annotations; red marks an assembly failure, not a confirmed annotation error.',font=small,fill='#555555')
        # Exact equality checks that annotation drawing never touches panel 1.
        assert canvas.crop((0,header,w,header+h)).tobytes()==original.tobytes()
        stem=f'{number:02d}_'+re.sub('[^a-zA-Z0-9_.-]+','_',asset)[-135:]
        filename=f'comparisons/{stem}.png';canvas.save(args.output/filename)
        (args.output/'annotations'/f'{stem}.json').write_bytes(annbytes)
        (args.output/'band_targets'/f'{stem}.json').write_bytes(targetbytes)
        thumb=canvas.copy();thumb.thumbnail((480,220));thumbnails.append((number,thumb))
        manifest.append(dict(number=number,asset_sample_id=asset,audit_sample_id=record['sample_id'],
            image=str(imagepath),annotation=str(annpath),band_target=str(targetpath),width=w,height=h,
            comparison=filename,failed_band_indices=sorted(bad),assembly_failures=[{k:c[k] for k in ['band_index','label','predicted','expected']} for c in cases],
            source_sha256={key:hashlib.sha256(payload).hexdigest() for key,payload in [('image',imagebytes),('annotation',annbytes),('band_target',targetbytes)]},
            alignment_error_px=audit_error))
        print('PACKAGED',number,'/56',asset,flush=True)
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
    with (args.output/'manifest.csv').open('w',newline='',encoding='utf-8-sig') as f:
        fields=['number','asset_sample_id','comparison','width','height','failed_band_indices','image','annotation','band_target']
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(manifest)
    sheet=Image.new('RGB',(4*480,14*250),'white');d=ImageDraw.Draw(sheet)
    for i,(number,thumb) in enumerate(thumbnails):
        x=(i%4)*480;y=(i//4)*250;sheet.paste(thumb,(x,y+22));d.text((x+4,y+2),f'{number:02d}',fill='black')
    sheet.save(args.output/'contact_sheet.jpg',quality=90)
    cards=[]
    for m in manifest:
        image=html.escape(m['comparison'],quote=True)
        cards.append(f'<article><h2>{m["number"]:02d} — {html.escape(m["asset_sample_id"])}</h2><p>Failed bands: {m["failed_band_indices"]}; native {m["width"]} × {m["height"]}</p><a href="{image}" target="_blank"><img src="{image}" loading="lazy"></a></article>')
    (args.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>56 assembly failure annotations</title><style>body{font:16px sans-serif;margin:24px}img{max-width:100%;height:auto}article{margin:36px 0;border-top:1px solid #ccc}h2{font-size:17px}</style><h1>56 张训练图：原图 | boxes＋原图 | bands＋原图</h1><p>点击图片查看原始分辨率。蓝色框：atom；品红框：charge。band 面板绘制所有真实 band，红色轮廓及 B 编号标出组图失败的 band；不表示已经确认标注有误。</p>'+''.join(cards))
    (args.output/'README.txt').write_text('56 张训练集组图失败样本检查包\n\n打开 index.html 可离线逐张浏览；comparisons/ 中恰好有 56 张横向三联 PNG。\n三列：训练原图 | 真实 atom/charge boxes＋原图 | 真实 bond bands＋原图。\n保留源图片原始分辨率；三列使用同一图片尺寸和坐标，没有拉伸或旋转。\n蓝框为原子，数字仅标出失败连接涉及的原子 ID，品红框为电荷。band 按类别着色；失败 band 额外用红轮廓及 B 编号标示，编号从 0 开始。\n红色表示几何组图与真实连接不一致，不代表已确认 annotation 错误。\n共 56 张独立图片、127 条失败 band；训练清单里的重复采样已去重。\nannotations/ 和 band_targets/ 保存对应的源标注副本；manifest.json 记录来源、失败连接、文件 SHA256 与坐标一致性检查。\n所有图片均通过源图尺寸、band 与标注一致性、640 训练坐标回溯检查；左列与源 RGB 图像像素完全一致。\n原始数据未修改。\n')
    archive=args.output.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in sorted(args.output.rglob('*')):
            if path.is_file():z.write(path,arcname=str(Path(args.output.name)/path.relative_to(args.output)))
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    assert len(list((args.output/'comparisons').glob('*.png')))==56
    print(json.dumps(dict(status='passed',images=56,failed_bands=127,archive=str(archive.resolve()),bytes=archive.stat().st_size)),flush=True)

if __name__=='__main__':main()
