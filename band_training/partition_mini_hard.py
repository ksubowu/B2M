"""Geometry-only quarantine of extreme miniature molecular drawings.

Scan first; application is an explicit second command. Prediction errors never
participate in selection. Original split/group and asset coordinate frames stay
unchanged. The apply implementation preserves frozen historical manifests.
"""
import argparse,csv,hashlib,html,json,math,os,shutil,time
from pathlib import Path
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from .mixed_data import letterbox_transform

ASSETS=['image','annotation','masks','band_target','svg','overlay','view_check','line_target','line_overlay','band_line_view_check']
RULE=dict(coordinate_frame='same rounded letterbox to 640x640 as training',roi_long_side_max=160.,median_atom_short_side_max=6.,median_bond_length_max=12.,
          expression='ROI long side <=160 AND median atom-box short side <=6 AND (median nonzero bond length <=12, or no bonds)',uses_predictions=False)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scan_one(args):
    root,record=args;ann=json.loads((root/record['annotation']).read_text());scale,offset,_=letterbox_transform(ann['width'],ann['height'])
    boxes=np.asarray([a['bbox'] for a in ann['atoms']],float).reshape(-1,2,2)*scale+offset
    span=boxes[:,1].max(0)-boxes[:,0].min(0);short=np.minimum(*(boxes[:,1]-boxes[:,0]).T)
    centers=np.asarray([a['center'] for a in ann['atoms']],float)*scale+offset
    lengths=np.array([np.linalg.norm(centers[b['source']]-centers[b['target']]) for b in ann['bonds']]);lengths=lengths[lengths>1e-8]
    median_length=float(np.median(lengths)) if len(lengths) else None
    selected=bool(max(span)<=RULE['roi_long_side_max'] and np.median(short)<=RULE['median_atom_short_side_max'] and (median_length is None or median_length<=RULE['median_bond_length_max']))
    return dict(sample_id=record['sample_id'],split=record['split'],variant=record['variant'],dataset=record['dataset'],atoms=len(ann['atoms']),charges=len(ann['charges']),
        roi_width=float(span[0]),roi_height=float(span[1]),roi_long_side=float(max(span)),median_atom_short_side=float(np.median(short)),
        median_bond_length=median_length,selected=selected)


def dump(path,obj):Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2))


def scan(root,out):
    out.mkdir(parents=True,exist_ok=False);records=json.loads((root/'data_manifest.json').read_text());started=time.time()
    with ThreadPoolExecutor(max_workers=8) as pool:metrics=list(pool.map(scan_one,[(root,r) for r in records]))
    selected={r['sample_id'] for r in metrics if r['selected']};subset=[r for r in records if r['sample_id'] in selected]
    known='bw/USPTO/US07323044-20080129-C00005'
    if any(r['sample_id']==known for r in records):assert known in selected,'Known 98x83 example must be selected'
    dump(out/'selection.json',dict(dataset=str(root.resolve()),data_manifest_sha256=sha(root/'data_manifest.json'),rule=RULE,records=subset,geometry=[r for r in metrics if r['selected']]))
    with (out/'all_geometry.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(metrics[0]));w.writeheader();w.writerows(metrics)
    counts=Counter(r['split'] for r in subset);report=dict(status='scanned',unique_images=len(records),selected=len(subset),split_counts=dict(counts),rule=RULE,seconds=time.time()-started)
    dump(out/'scan_report.json',report)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',12)
    preview=Image.new('RGB',(1280,260*max(1,math.ceil(min(32,len(subset))/4))),'white');draw=ImageDraw.Draw(preview)
    # Largest selected footprint first: visually inspect the inclusion boundary.
    ordered=sorted([m for m in metrics if m['selected']],key=lambda x:-x['roi_long_side'])
    lookup={r['sample_id']:r for r in records}
    for i,m in enumerate(ordered[:32]):
        with Image.open(root/lookup[m['sample_id']]['image']) as image:
            rgb=image.convert('RGB');rgb.thumbnail((310,220));x=(i%4)*320;y=(i//4)*260;preview.paste(rgb,(x+(320-rgb.width)//2,y))
        draw.text((x+3,y+220),f"{i} {m['split']} ROI {m['roi_width']:.0f}x{m['roi_height']:.0f} atom {m['median_atom_short_side']:.1f}px",font=font,fill='black')
        draw.text((x+3,y+236),m['sample_id'][-40:],font=font,fill='black')
    preview.save(out/'boundary_preview.jpg');dump(out/'preview_order.json',ordered[:32])
    print(json.dumps(report,indent=2),flush=True)


def write_csv(path,records,fields=None):
    if fields is None:fields=list(dict.fromkeys(k for r in records for k in r)) or ['sample_id','split','image','annotation','SMILES','source_SMILES']
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader()
        for r in records:
            writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else '' if v is None else v for k,v in r.items()})


def atomic_json(path,records):
    temp=path.with_name(path.name+'.partition_tmp');dump(temp,records);os.replace(temp,path)


def index_page(path,records,title):
    brief=[{k:r.get(k,'') for k in ['sample_id','split','dataset','SMILES','image','annotation','band_line_view_check']} for r in records]
    payload=json.dumps(brief,ensure_ascii=False).replace('<','\\u003c')
    page='''<!doctype html><meta charset="utf-8"><title>TITLE</title><style>body{font:15px sans-serif;margin:24px;background:#f5f7fa}input{width:70%;padding:10px}article{background:white;margin:14px 0;padding:14px;border-radius:8px}img{height:250px;max-width:65%;object-fit:contain}small{display:block;overflow-wrap:anywhere}</style><h1>TITLE</h1><p>SIZE unique images. Original splits preserved. <a href="smiles.csv">SMILES CSV</a> · <a href="data_manifest.json">manifest</a></p><input id="q" placeholder="sample ID / dataset / SMILES"><button id="prev">Previous</button><button id="next">Next</button><span id="count"></span><main id="list"></main><script>
const data=PAYLOAD;let page=0;function render(){let q=document.getElementById('q').value.toLowerCase(),rows=data.filter(r=>(r.sample_id+' '+r.dataset+' '+r.SMILES).toLowerCase().includes(q));let limit=Math.max(0,Math.ceil(rows.length/24)-1);page=Math.min(page,limit);document.getElementById('count').textContent=` ${rows.length} results · page ${page+1}`;let box=document.getElementById('list');box.replaceChildren();for(let r of rows.slice(page*24,page*24+24)){let card=document.createElement('article'),h=document.createElement('h3');h.textContent=r.sample_id+' ['+r.split+']';card.append(h);for(let key of ['image','band_line_view_check']){if(!r[key])continue;let a=document.createElement('a');a.href=r[key];let im=document.createElement('img');im.loading='lazy';im.src=r[key];a.append(im);card.append(a)}let text=document.createElement('small');text.textContent=r.SMILES;card.append(text);let a=document.createElement('a');a.href=r.annotation;a.textContent='Annotation JSON';card.append(a);box.append(card)}}document.getElementById('q').oninput=()=>{page=0;render()};document.getElementById('prev').onclick=()=>{page=Math.max(0,page-1);render()};document.getElementById('next').onclick=()=>{page++;render()};render();</script>'''
    Path(path).write_text(page.replace('TITLE',html.escape(title)).replace('SIZE',str(len(records))).replace('PAYLOAD',payload))


def basic_stats(records):
    truth=lambda v:v is True or str(v).lower() in ['1','true','yes']
    return dict(images=len(records),charged_images=sum(truth(r.get('has_charge')) for r in records),
                color=sum(truth(r.get('is_color')) for r in records),by_variant=dict(Counter(r['variant'] for r in records)),by_dataset=dict(Counter(r['dataset'] for r in records)))


def apply_partition(root,out):
    choice=json.loads((out/'selection.json').read_text());selected={r['sample_id'] for r in choice['records']};mini=root/'mini_hard'
    if (root/'mini_hard_partition_report.json').exists():
        old=json.loads((root/'mini_hard_partition_report.json').read_text())
        assert set(old['selected_ids'])==selected and old['status']=='completed';print('Partition already applied');return
    assert sha(root/'data_manifest.json')==choice['data_manifest_sha256'],'Dataset changed after selection'
    assert not mini.exists(),'mini_hard already exists'
    backup=root/'_before_mini_hard_20261121';assert not backup.exists(),'Backup directory already exists'
    original=json.loads((root/'data_manifest.json').read_text());normal=[r for r in original if r['sample_id'] not in selected]
    canonical=lambda r:r.get('asset_sample_id') or r['sample_id'].split('::repeat_')[0]
    stems=['data_manifest','unique_train','train','val','source_train','charge_manifest','render_manifest']
    manifests={name:json.loads((root/(name+'.json')).read_text()) for name in stems}
    originals={p.name:p for p in root.iterdir() if p.is_file() and not p.name.startswith('.')}
    lock=root/'.mini_hard_partition.lock';lock_fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    operations=[];changed=[];report=None
    try:
        backup.mkdir()
        for name,path in originals.items():shutil.copy2(path,backup/name)
        dump(backup/'files_sha256.json',{name:sha(path) for name,path in originals.items()})
        mini.mkdir();normal_paths={r[k] for r in normal for k in ASSETS if r.get(k)}
        paths=sorted({r[k] for r in choice['records'] for k in ASSETS if r.get(k)})
        # Old locations remain resolvable for frozen historical reports. Shared
        # assets stay in normal data and receive a verified copy in mini_hard.
        for relative in paths:
            src=root/relative;dest=mini/relative
            assert src.is_file() and src.resolve().is_relative_to(root.resolve()),relative
            digest=sha(src);dest.parent.mkdir(parents=True,exist_ok=True)
            if relative in normal_paths:
                shutil.copy2(src,dest);mode='copy_shared'
            else:
                os.replace(src,dest);src.symlink_to(os.path.relpath(dest,src.parent));mode='move_with_compatibility_link'
            operations.append(dict(path=relative,mode=mode,sha256=digest))
            assert sha(dest)==digest and sha(src)==digest,relative
        for name,records in manifests.items():
            regular=[r for r in records if canonical(r) not in selected];hard=[r for r in records if canonical(r) in selected]
            atomic_json(root/(name+'.json'),regular);changed.append(name+'.json');dump(mini/(name+'.json'),hard)
            csv_path=root/(name+'.csv')
            fields=None
            if csv_path.exists():
                with (backup/(name+'.csv')).open(newline='') as f:
                    reader=csv.DictReader(f);fields=reader.fieldnames;csv_regular=[r for r in reader if canonical(r) not in selected]
                temp=csv_path.with_name(csv_path.name+'.partition_tmp');write_csv(temp,csv_regular,fields);os.replace(temp,csv_path);changed.append(name+'.csv')
            write_csv(mini/(name+'.csv'),hard,fields)
        normal_train=[r for r in normal if r['split']=='train'];normal_val=[r for r in normal if r['split']=='val']
        hard_train=[r for r in choice['records'] if r['split']=='train'];hard_val=[r for r in choice['records'] if r['split']=='val']
        current_sampling=json.loads((root/'train.json').read_text());hard_sampling=json.loads((mini/'train.json').read_text())
        for base,data,training,validation,sampling in [(root,normal,normal_train,normal_val,current_sampling),(mini,choice['records'],hard_train,hard_val,hard_sampling)]:
            config=json.loads((backup/'dataset_config.json').read_text());config.update(unique_train_images=len(training),validation_images=len(validation),train_sampling_rows=len(sampling),
                unique_validation_fraction=len(validation)/len(data) if data else 0,source_valid_images=sum(r['variant']=='bw' for r in data),
                base_color_variants=sum(r['variant']=='color' for r in data),training_charge_variants=sum(r['variant']=='charge_augmented' for r in data),
                partition=dict(name='mini_hard' if base==mini else 'normal',selection_rule=choice['rule'],historical_full_dataset_backup=backup.name))
            config['charge_balance']['actual_charged_fraction']=basic_stats(sampling)['charged_images']/len(sampling) if sampling else 0.
            if base==mini:config['charge_balance']['enabled']=False
            atomic_json(base/'dataset_config.json',config)
            atomic_json(base/'distribution_report.json',dict(all_unique=basic_stats(data),train_unique=basic_stats(training),validation=basic_stats(validation),train_sampling=basic_stats(sampling),partition_rule=choice['rule']))
            atomic_json(base/'split_report.json',dict(train_images=len(training),validation_images=len(validation),train_val_group_overlap=len({r['group'] for r in training}&{r['group'] for r in validation}),source_split_preserved=True))
            atomic_json(base/'balance_report.json',dict(sampling_rows=len(sampling),unique_training_images=len(training),charged_rows=basic_stats(sampling)['charged_images'],actual_charged_fraction=config['charge_balance']['actual_charged_fraction'],sampling_weights_preserved=True))
            smiles=[{k:r.get(k,'') for k in ['sample_id','split','dataset','SMILES','source_SMILES','image','annotation','band_target','line_target','masks','svg']} for r in data]
            write_csv(base/'smiles.csv',smiles)
            index_page(base/'index.html',data,'mini_hard — extreme miniature molecular drawings' if base==mini else 'Normal molecular dataset — mini_hard excluded')
        # Reuse optional renderer runtime without duplicating its source bundle.
        (mini/'generator_snapshot').symlink_to('../generator_snapshot',target_is_directory=True)
        shutil.copy2(out/'selection.json',mini/'selection.json');shutil.copy2(out/'boundary_preview.jpg',mini/'preview.jpg')
        checks=dict(normal_images=len(normal),mini_hard_images=len(choice['records']),total_images=len(original),files_checked=0)
        for base,records in [(root,normal),(mini,choice['records'])]:
            for r in records:
                for k in ASSETS:
                    if r.get(k):assert (base/r[k]).is_file(),(base,r[k]);checks['files_checked']+=1
        assert len(normal)+len(choice['records'])==len(original)
        assert not ({r['sample_id'] for r in normal}&selected)
        assert not ({r['group'] for r in normal_train}&{r['group'] for r in normal_val})
        assert not ({r['group'] for r in hard_train}&{r['group'] for r in hard_val})
        report=dict(status='completed',selected_ids=sorted(selected),rule=choice['rule'],backup=str(backup),asset_operations=operations,
            original=basic_stats(original),normal=basic_stats(normal),mini_hard=basic_stats(choice['records']),normal_train=len(normal_train),normal_val=len(normal_val),
            mini_train=len(hard_train),mini_val=len(hard_val),normal_train_sampling_rows=len(current_sampling),mini_train_sampling_rows=len(hard_sampling),
            train_val_groups_preserved=True,checks=checks,history='Historical evaluation reports remain full-corpus results. New normal manifests define a different cohort. Restoring backed-up manifests restores full-corpus access through compatibility links.')
        validation=dict(status='passed',scope='Partition membership, paths and byte-identical archived assets. Original chemistry/ownership audit inherited without editing annotations.',checks=checks,
            inherited_validation=str(backup/'validation_report.json'))
        atomic_json(root/'validation_report.json',validation);dump(mini/'validation_report.json',validation)
        annotation_convention=(backup/'README.md').read_text().split('## 标注约定')[1].split('## 查看与验证')[0]
        root_readme=f'''# 常规化学结构数据（mini_hard 已分离）

2026-11-21 按纯几何规则归档极端小图；未使用模型识别结果筛选。图片和标注坐标均未修改。

- 常规唯一图像：{len(normal)}；唯一训练图：{len(normal_train)}；验证图：{len(normal_val)}。
- 默认 `train.csv/json` 仍使用原采样权重，共 {len(current_sampling)} 行；去重评估用 `unique_train.csv/json`。
- `mini_hard/`：{len(choice['records'])} 张（原 train {len(hard_train)}、原 val {len(hard_val)}），常规清单已排除。
- 阈值按训练使用的 640×640 letterbox 坐标计算：分子区域长边≤160 px，atom box短边中位数≤6 px，非零键长中位数≤12 px（无键图只检查前两项）。
- SMILES 与源 SMILES 见 `smiles.csv`；原有 annotation、mask、band、line、SVG 均保留。
- 实体文件移入 mini_hard；旧路径保留兼容链接，旧报告仍可查看。
- 原全量清单、README、配置和报告逐字节备份于 `{backup.name}/`。此前14,980图的评估仍为历史全量口径；新常规结果应注明已排除mini_hard。
- 生成阶段日志/检查点记录的是原始全量生成任务；当前训练与评估以本目录的新清单和 `mini_hard_partition_report.json` 为准。重新生成数据请使用新输出目录。

## 标注约定
{annotation_convention}
'''
        (root/'README.md').write_text(root_readme)
        (mini/'README.md').write_text(f'''# mini_hard：极端小尺寸分子图

共{len(choice['records'])}张，保留原划分：train={len(hard_train)}，val={len(hard_val)}。用于后续小目标专项模型准备，本次未训练模型。

- `data_manifest.json/csv`：全部唯一图像；`unique_train.json/csv`、`train.json/csv` 和 `val.json/csv` 保留原分组。
- `smiles.csv` 同时保存实际绘制SMILES和source_SMILES；annotation JSON 内另含完整rendered_molfile、atom/charge boxes、bond bands和lines。
- `assets/` 下实体文件保持原相对路径；PNG、annotation、mask、band target、line target、SVG及检查叠加图完整保留，未缩放或改标注。
- `selection.json` 保存规则和每张图的几何量；`preview.jpg`、`index.html` 方便浏览。
- `generator_snapshot` 是指向上级共享生成器的本地链接；数据本体无此依赖。迁移并运行Indigo相关评估时一并带上共享生成器。
- 过滤规则：640×640输入中的ROI长边≤160、atom框短边中位数≤6、非零键长中位数≤12。规则仅用GT几何，不用模型分数、错误或OCR结果。
''')
        dump(root/'mini_hard_partition_report.json',report);dump(mini/'partition_report.json',report);dump(out/'applied_report.json',report)
        print(json.dumps({k:v for k,v in report.items() if k!='asset_operations'},indent=2),flush=True)
    except Exception:
        # Keep an interrupted organization operation reversible and preserve all
        # original data; restore manifests and actual asset locations.
        if backup.exists():
            for name in originals:
                if (backup/name).exists():shutil.copy2(backup/name,root/name)
        for item in reversed(operations):
            src=root/item['path'];dest=mini/item['path']
            if item['mode']=='move_with_compatibility_link':
                if src.is_symlink():src.unlink()
                if dest.exists():os.replace(dest,src)
        for name in ['smiles.csv','mini_hard_partition_report.json','distribution_report.json','split_report.json','balance_report.json','validation_report.json']:
            if name not in originals and (root/name).is_file():(root/name).unlink()
        if mini.exists():shutil.rmtree(mini)
        raise
    finally:
        os.close(lock_fd);lock.unlink()


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--report',type=Path,required=True)
    p.add_argument('--apply',action='store_true');a=p.parse_args()
    if a.apply:apply_partition(a.dataset,a.report)
    else:scan(a.dataset,a.report)


if __name__=='__main__':main()
