"""Validate duplicate-candidate diagnosis and export original/before/after crops."""
import argparse,gzip,json,math,csv,html,zipfile,shutil
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageOps
from .mixed_data import ATOM_CLASSES,letterbox_transform
from .geometry import edge_key,target_edges,polygon_iou
from .train_full_v3 import metric_row
from .continue_indigo import write

NAMES=['single','double','triple','aromatic','wedge','dash']


def render_crop(dataset,d,predids,path,title,gt_type=None):
    with Image.open(dataset/d['record']['image']) as image:original=image.convert('RGB')
    _,offset,shape=letterbox_transform(*original.size)
    canvas=Image.new('RGB',(640,640),'white');canvas.paste(original.resize(shape,Image.Resampling.LANCZOS),tuple(offset))
    coordinates=np.concatenate([np.array(d['instances'][i]['quad']).reshape(-1,2) for i in predids])
    x1,y1=np.maximum(0,np.floor(coordinates.min(0)-35)).astype(int);x2,y2=np.minimum(640,np.ceil(coordinates.max(0)+35)).astype(int)
    panels=[canvas.crop((x1,y1,x2,y2)) for _ in range(3)];pairids={}
    for edge in d['bonds']:pairids[edge['bond_instance']]=tuple(sorted((edge['begin'],edge['end'])))
    winner=max(predids,key=lambda i:d['instances'][i]['score'])
    for panel,ids in [(panels[1],predids),(panels[2],[winner])]:
        draw=ImageDraw.Draw(panel)
        for i in ids:
            item=d['instances'][i];color='green' if gt_type is not None and item['label']==gt_type else 'red'
            quad=[(float(x-x1),float(y-y1)) for x,y in item['quad']];line=[(float(x-x1),float(y-y1)) for x,y in item['line']]
            draw.line(quad+[quad[0]],fill=color,width=2);draw.line(line,fill='blue',width=1)
            for x,y in line:draw.ellipse((x-2,y-2,x+2,y+2),outline='purple',width=1)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
    out=Image.new('RGB',(1260,450),'white');draw=ImageDraw.Draw(out)
    for col,(panel,label) in enumerate(zip(panels,['Original crop','Current kept candidates','One bond per atom pair'])):
        im=ImageOps.contain(panel,(400,285));out.paste(im,(col*420+10,50));draw.text((col*420+10,12),label,font=font,fill='black')
    draw.text((10,345),title,font=font,fill='black')
    draw.text((10,370),'Candidates: '+'; '.join(f'{i}: {NAMES[d["instances"][i]["label"]]} {d["instances"][i]["score"]:.3f}' for i in predids),font=font,fill='black')
    draw.text((10,395),'GT: '+(NAMES[gt_type] if gt_type is not None else 'see annotation')+' | '+d['sample_id'],font=font,fill='black')
    out.save(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--diagnosis',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);p.add_argument('--dataset',type=Path,required=True)
    args=p.parse_args();out=args.diagnosis;report=json.loads((out/'report.json').read_text());rows=json.loads((out/'samples.json').read_text())
    byid={r['sample_id']:r for r in rows}
    with gzip.open(args.cache,'rt') as f:data=[json.loads(l) for l in f]
    stats=Counter();confusion=Counter();scores=defaultdict(list);pairious=[];combined=defaultdict(list);examples=[];regression_edges=[]
    preview=out/'examples';preview.mkdir(exist_ok=True)
    for d in data:
        r=byid[d['sample_id']];old=r['original'];new=r['post_dedup'];mapping={int(i):j for i,j in d['atom_mapping'].items()}
        truth=target_edges(d['ann']);gtpair={tuple(sorted((k[0],k[1]))):k for k in truth}
        links={e['bond_instance']:e for e in d['bonds']};groups=defaultdict(list)
        for i,e in links.items():groups[tuple(sorted((e['begin'],e['end'])))].append(i)
        stats.update(dedup_fixed=int(not old['exact'] and new['exact']),dedup_regressed=int(old['exact'] and not new['exact']),
            duplicate_pair_images=int(any(len(g)>1 for g in groups.values())))
        if not old['exact']:stats['failure_fp_only' if old['fp'] and not old['fn'] else 'failure_fn_only' if old['fn'] and not old['fp'] else 'failure_both']+=1
        valid_pred=Counter(edge_key(mapping.get(e['begin'],-1),mapping.get(e['end'],-1),e['label']) for e in links.values() if min(mapping.get(e['begin'],-1),mapping.get(e['end'],-1))>=0)
        for x in r['extra']:
            if x['category']=='wrong_bond_class':
                key=gtpair[tuple(sorted(x['mapped_edge'][:2]))]
                present=valid_pred[key]>0
                category='extra_competing_class_with_correct_bond_present' if present else 'true_class_confusion_without_correct_class'
                stats[category]+=1;confusion[NAMES[key[2]]+' -> '+NAMES[x['label']]]+=1;scores[category].append(x['score'])
            elif x['category']=='unmatched_or_wrong_class_atom':
                e=links[x['prediction']]
                for i in [e['begin'],e['end']]:
                    if i not in mapping:stats['unmatched_atom_'+ATOM_CLASSES[d['atoms'][i]['label']]]+=1
        for pair,ids in groups.items():
            if len(ids)<2:continue
            for i in range(len(ids)):
                for j in range(i):pairious.append(polygon_iou(d['instances'][ids[i]]['quad'],d['instances'][ids[j]]['quad']))
            u,v=(mapping.get(i,-1) for i in pair);key=gtpair.get(tuple(sorted((u,v))))
            winner=max(ids,key=lambda i:d['instances'][i]['score'])
            if key and valid_pred[key] and edge_key(mapping.get(links[winner]['begin'],-1),mapping.get(links[winner]['end'],-1),d['instances'][winner]['label'])!=key:
                regression_edges.append(dict(sample_id=d['sample_id'],pair=pair,gt=key,candidates=[dict(index=i,label=d['instances'][i]['label'],score=d['instances'][i]['score']) for i in ids]))
            if not old['exact'] and new['exact'] and key and len(examples)<12 and len(d['ann']['bonds'])<=60:
                example=dict(sample_id=d['sample_id'],pair=pair,gt_type=NAMES[key[2]],candidates=[dict(index=i,label=NAMES[d['instances'][i]['label']],score=d['instances'][i]['score']) for i in ids],
                    quad_iou=polygon_iou(d['instances'][ids[0]]['quad'],d['instances'][ids[1]]['quad']))
                filename=f'duplicate_{len(examples):02d}.png';render_crop(args.dataset,d,ids,preview/filename,'Duplicate candidates retained by quad NMS',key[2]);example['image']='examples/'+filename;examples.append(example)
        for threshold in [.2,.3,.4,.5]:
            keep=[];seen=set()
            for i in sorted(links,key=lambda i:-d['instances'][i]['score']):
                if d['instances'][i]['score']<threshold:continue
                e=links[i];pair=tuple(sorted((e['begin'],e['end'])))
                if pair not in seen:seen.add(pair);keep.append(i)
            keep += [i for i in d['unlinked'] if d['instances'][i]['score']>=threshold]
            metric,_=metric_row([d['instances'][i] for i in sorted(keep)],d['ann'],d['atoms'],mapping,'forward',0.)
            combined[str(threshold)].append(metric)
    from .geometry import aggregate
    stats['correct_edges_lost_by_highest_score_choice']=sum(r['original']['tp']-r['post_dedup']['tp'] for r in rows if r['post_dedup']['tp']<r['original']['tp'])
    extra=dict(counts=dict(stats),wrong_class_confusions=dict(confusion),duplicate_quad_iou=dict(n=len(pairious),median=float(np.median(pairious)),p90=float(np.quantile(pairious,.9))),
        confidence={k:dict(n=len(v),median=float(np.median(v)),p90=float(np.quantile(v,.9))) for k,v in scores.items()},
        combined_threshold_dedup={k:aggregate(v) for k,v in combined.items()},no_production_changes=True)
    write(out/'duplicate_analysis.json',extra);write(out/'duplicate_examples.json',examples);write(out/'highest_score_wrong_choice.json',regression_edges)
    c=report['counts'];fp=report['fp_categories'];oracle=report['oracles'];dedup=oracle['one_bond_per_predicted_atom_pair']
    text=f'''键预测失败诊断：1498张验证图，冻结同一权重和原阈值0.2

当前严格键图全对1195/1498=79.77%；303张失败。
真实36545条键，TP=36135，FP=804，FN=410。FP/FN是typed edge误差，错误类别/连接可能同时产生一条FP和一条FN。
失败图：仅多报{stats['failure_fp_only']}张，仅漏/错键而无多报{stats['failure_fn_only']}张，两类同时有{stats['failure_both']}张。
其中241张原子全部匹配正确但键图错误；62张同时存在原子检测/类别/匹配问题。

主要原因：同一对原子保留了多个竞争候选。
{stats['duplicate_pair_images']}张图存在重复原子对候选，共{c['duplicate_atom_pair_candidates']}个可被pair去重的额外候选。
FP中的wrong_bond_class有{fp['wrong_bond_class']}条，其中{stats['extra_competing_class_with_correct_bond_present']}条发生在正确键已存在时；它们主要是额外类别候选，不是该真实键完全认错类别。
独立的6通道sigmoid heatmap可同时对不同键型给高分。现有suppress仅按quad IoU>=0.45去重；同一物理键的窄/宽候选可能IoU很小。
实测重复原子对候选的quad IoU中位数{extra['duplicate_quad_iou']['median']:.3f}，90%分位{extra['duplicate_quad_iou']['p90']:.3f}；具体例子见examples/。
最终没有执行“一对原子只能有一条键”的约束。

FP分解（互斥且合计804）：
'''+''.join(f'{k}: {v}\n' for k,v in fp.items())+f'''
295个未参与正确计分关联的候选中：89个确实未入图（主要两端落同一个预测atom），206个已连接到预测原子但至少一端无法正确匹配GT。
只比较实际组装出的typed edge集合、忽略被拒绝候选时，键图全对为1206/1498；正式模型输出指标仍对未关联候选计错。

不改权重的验证集后处理对照（未替换生产推理）：
每对预测原子只保留最高置信度一条键：{dedup['exact']}/1498={100*dedup['graph_exact']:.2f}%，修复{stats['dedup_fixed']}张、原全对图退步{stats['dedup_regressed']}张。
此时TP={dedup['tp']}、FP={dedup['fp']}、FN={dedup['fn']}；有{stats['correct_edges_lost_by_highest_score_choice']}条原本已正确的键候选被更高分错误类型挤掉，仍需置信度/类别校准。
只把阈值升到0.4：1334/1498=89.05%，可见原0.2阈值偏宽松，但继续升高会明显漏检。
以上均是当前验证集对照，不是新训练或独立测试结果。

其它原因：
真实键缺失或区域错位：诊断几何配对中148条GT键没有匹配到当前阈值后的预测区域；需结合图像核对，不能把此数一概解释为网络没看到键。
132条有对应区域但涉及原子检测/匹配错误；68条区域有对应却连错端点；36条区域有对应但line未能关联。
24条GT键的原子对正确、类别不对；2条是立体方向错误。
几何配对使用与类别无关的区域Hungarian匹配，IoU>=0.1或严格中心/端点距离；这些FN归因和oracle替换只是诊断，不改变正式评分。

诊断对照：预测line改查真实atom锚点，全对1213/1498=80.97%；改善有限，说明大部分问题不是单纯box中心偏移。
给所有真实line与类别，仍用预测atom boxes：1447/1498=96.60%。
真实line+真实atom boxes用当前forward KD-tree：1487/1498=99.27%，还有11张因框中心/锚点关系而失败。

按最终连接计的键型F1：单键98.37%、双键98.83%、三键90.76%、芳香键97.62%、楔形88.69%、虚楔形89.59%。
稀有键型召回仍较高（95.80%—97.41%），精度偏低是更明显的问题。
验证集三键116、楔形203、虚楔形238条，对比单键25589、双键10109条，类别明显长尾。长尾可能影响置信度校准，本次没有用训练消融证明因果。
复杂度分层：<=20条键569/751=75.77%，21—40条467/539=86.64%，41—80条159/205=77.56%，>80条0/3。
整体错误不随键数单调上升：小分子也常因额外候选整图失败；最大分子确有漏检/端点错误，但>80条只有3张，不能过度外推。

优先改进：
1. 推理增加原子对唯一性约束，并评估line端点相似度NMS，避免仅依赖band quad IoU。
2. 校准阈值及竞争类别置信度；不能盲目用最高分永远替换，20条受损边需要保护。
3. 训练上加强跨类别重复候选抑制和单实例键型判别，保留atom/charge与bond预测解耦。
4. 针对短键、密集大分子和少数键型处理剩余漏检/端点错误。>80条键的样本只有3张，不能据此概括全部复杂分子。
'''
    (out/'report.txt').write_text(text)
    with (out/'failed_samples.csv').open('w',newline='') as f:
        fields=['sample_id','atoms','bonds','atom_exact','tp','fp','fn','unlinked','dedup_exact'];writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in rows:
            if not r['bond_exact']:writer.writerow(dict(sample_id=r['sample_id'],atoms=r['atoms'],bonds=r['bonds'],atom_exact=r['atom_exact'],
                **{k:r['original'][k] for k in ['tp','fp','fn','unlinked']},dedup_exact=r['post_dedup']['exact']))
    (out/'index.html').write_text('<meta charset="utf-8"><h1>Bond failure diagnosis</h1><p><a href="report.txt">完整报告</a> | <a href="failed_samples.csv">303张失败样本</a></p>'+''.join('<article><p>'+html.escape(e['sample_id']+' GT '+e['gt_type'])+'</p><img width="100%" src="'+e['image']+'"></article>' for e in examples))
    src=out/'source';src.mkdir(exist_ok=True)
    for name in ['diagnose_bond_failures.py','review_bond_failures.py']:shutil.copyfile(Path(__file__).with_name(name),src/name)
    with zipfile.ZipFile(out.with_suffix('.zip'),'w',zipfile.ZIP_DEFLATED) as z:
        for path in out.rglob('*'):
            if path.is_file():z.write(path,path.relative_to(out.parent))
    print(json.dumps(extra,indent=2),flush=True)

if __name__=='__main__':main()
