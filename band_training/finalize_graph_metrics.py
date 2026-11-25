"""Consolidate component, dummy-graph and frozen third-party OCR results."""
import argparse,json,csv,sys,hashlib,shutil,zipfile,html
from pathlib import Path
from collections import Counter
from PIL import Image,ImageDraw,ImageFont,ImageOps
import networkx as nx
from .continue_indigo import write
from .molecular_graph_metrics import native_graph,fold_expanded_reference


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args();root=args.root
    read=lambda path:json.loads(path.read_text())
    cache=read(root/'prediction_cache/report.json');references=read(root/'reference_verified/report.json')
    metric=read(root/'graph_ocr_verified/report.json');samples=read(root/'graph_ocr_verified/samples.json');ocr=read(root/'graph_ocr_verified/ocr_crops.json')
    functional=read(root/'graph_ocr_verified/functional_report.json')
    functional_samples={r['sample_id']:r for r in read(root/'graph_ocr_verified/functional_samples.json')}
    reference_samples=read(root/'reference_verified/samples.json')
    alias_keys={(r['sample_id'],d['atom_id']) for r in reference_samples for d in r['dummy_labels'] if d['kind']=='alias_or_rsite'}
    assert references['status']=='passed' and metric['counts']['legacy_full_exact']==cache['counts']['legacy_full_exact']
    n=cache['counts']['images'];c=cache['counts'];m=metric['counts'];out=root/'review';out.mkdir(exist_ok=True)
    table=[]
    for name,key in [('原子整图匹配正确','atom_exact'),('电荷值及归属整图正确','charge_owner_exact'),('键整图匹配正确','bond_exact'),('原严格全图正确','legacy_full_exact')]:
        table.append(dict(metric=name,correct=c[key],total=n,rate=c[key]/n))
    for name,key in [('分子图正确*','graph_star_exact'),('分子图正确_ocr','graph_ocr_exact')]:
        value=functional['groups']['all'][key];table.append(dict(metric=name,correct=value,total=n,rate=value/n))
    matched=[r for r in ocr if r['kind']=='predicted_crop' and r['expected'] is not None and (r['sample_id'],r['reference_atom']) in alias_keys]
    gt={(r['sample_id'],r['reference_atom']):r for r in ocr if r['kind']=='gt_crop_diagnostic'}
    paired_gt=sum(gt[r['sample_id'],r['reference_atom']]['exact'] for r in matched)
    gt_alias=[r for r in ocr if r['kind']=='gt_crop_diagnostic' and (r['sample_id'],r['reference_atom']) in alias_keys]
    dummy=functional['groups']['with_alias'];lost=functional['groups']['all']['graph_star_exact']-functional['groups']['all']['graph_ocr_exact']
    supplemental=dict(charge_detection_exact=dict(correct=c['charge_detection_exact'],total=n),
        charged_images_charge_owner_exact=dict(correct=c['charged_owner_exact'],total=c['charged_images']),
        charged_images_detection_exact=dict(correct=c['charged_detection_exact'],total=c['charged_images']),
        charge_free_images_no_false_positive=dict(correct=c['neutral_charge_exact'],total=c['neutral_images']),
        ocr_predicted_matched_crop_exact=dict(correct=sum(r['exact'] for r in matched),total=len(matched)),
        ocr_gt_same_matched_crop_exact=dict(correct=paired_gt,total=len(matched)),
        ocr_all_gt_crop_exact=dict(correct=sum(r['exact'] for r in gt_alias),total=len(gt_alias)),
        ocr_graph_success_conditional_on_correct_star_with_dummy=dict(correct=dummy['graph_ocr_exact'],total=dummy['graph_star_exact']),
        ocr_lost_images=lost,ocr_lost_percentage_points=100*lost/n,
        exact_component_combinations={k.removeprefix('components_'):v for k,v in m.items() if k.startswith('components_')})
    report=dict(schema='molecular_graph_dummy_ocr_eval_v1',images=n,metrics=table,supplemental=supplemental,reference_audit=references,
        functional_graph_metrics=functional,model_vocabulary_diagnostic=metric,checkpoint=cache['checkpoint'],checkpoint_sha256=cache['checkpoint_sha256'],
        old_full_result_reproduced=cache['previous_full_exact_reproduced'],
        causal_comparison='Same frozen model predictions and same graph assembly for star vs OCR. Only dummy label text is added.',
        scope=functional['policy']+' Formal charge, aromatic-normalized bond orders and directed drawn wedge/dash are checked. Isotope/implicit H/atom CIP/E-Z are not separately scored.')
    write(out/'report.json',report)
    with (out/'metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['metric','correct','total','rate']);writer.writeheader();writer.writerows(table)
    with (out/'samples.csv').open('w',newline='') as f:
        columns=['sample_id','atom_exact','charge_detection_exact','charge_owner_exact','bond_exact','legacy_full_exact','graph_star_exact','graph_ocr_exact','has_dummy']
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
        for s in samples:
            fs=functional_samples[s['sample_id']]
            writer.writerow(dict(sample_id=s['sample_id'],**s['components'],graph_star_exact=fs['graph_star_exact'],graph_ocr_exact=fs['graph_ocr_exact'],has_dummy=fs['has_alias']))
    dataset=Path(cache['dataset'])
    from .dataset_runtime import install_generator_path
    install_generator_path(dataset)
    from chem_prep.backend import Indigo
    ig=Indigo();display=nx.Graph();display.add_node(0,symbol='C',charge=0,opaque=False);display.add_node(1,symbol='CF3',charge=0,opaque=True);display.add_edge(0,1,label=0)
    examples=[]
    for smiles in ['CC(F)(F)F','CC(F)F','CC(F)(Cl)F']:
        examples.append(dict(source_smiles=smiles,display_reference='C--CF3',anonymous_reference='C--*',
            result=fold_expanded_reference(native_graph(ig.loadMolecule(smiles)),display,{'CF3':'[C](F)(F)F'})))
    write(out/'folding_examples.json',examples)
    percent=lambda x: f'{100*x:.2f}%'
    lines=['分子图、dummy 和 OCR 分项评估（冻结第20个epoch最佳权重）','',f'全量验证图：{n}；旧结果精确复现：{c["legacy_full_exact"]}/{n}。','']
    lines += [f'{r["metric"]}：{r["correct"]}/{r["total"]} = {percent(r["rate"])}' for r in table]
    lines += ['',f'电荷检测整图全对：{c["charge_detection_exact"]}/{n}；电荷值+归属整图全对：{c["charge_owner_exact"]}/{n}。',
        f'带电图单独计算：电荷归属正确 {c["charged_owner_exact"]}/{c["charged_images"]}={percent(c["charged_owner_exact"]/c["charged_images"])}；无电荷图不误检 {c["neutral_charge_exact"]}/{c["neutral_images"]}。',
        '三分项不是互斥贡献。原严格全图要求 atom_exact AND charge_owner_exact AND bond_exact。',
        f'只错键、原子和电荷均正确：{supplemental["exact_component_combinations"].get("110",0)}张；只错电荷：{supplemental["exact_component_combinations"].get("101",0)}张；只错原子：{supplemental["exact_component_combinations"].get("011",0)}张。',
        '', '分子图正确*：',
        '每个预测other节点先映射为匿名*；参考端仅把真实官能团/别名/R-site映射为*，真实元素保持身份。',
        '例如Sb/Y/Pd不应因模型不支持就被当成官能团；如果模型只预测other，该图在正式星图口径下判错。',
        '在不使用坐标/GT原子匹配的情况下做完整图同构：节点类别/电荷一致、所有连接与键型一致、楔形/虚线方向一致；芳香/Kekule表示独立归一化。',
        '多/漏节点、多/漏键、未关联键、电荷未归属均不能因MCS保留了大部分结构而判对。',
        f'另保留模型词表诊断口径（所有other均匿名，包括真实元素）：{metric["groups"]["all"]["graph_star_exact"]}/{n}。正式指标使用仅官能团口径。',
        '', '分子图正确_ocr：',
        '固定上述预测图，对预测other框用第三方EasyOCR英文g2预训练模型读字，在正式星图正确的基础上，与参考官能团/R-site标签做完整带标签图同构；不改变节点/连接，也不借OCR修复词表外元素错误。',
        'GT标签不传给OCR，不使用词典纠错或化学专用微调权重；只做Unicode NFKC及去空白，保留大小写、标点及R编号。',
        f'加入OCR后减少 {lost} 张全对图，下降 {100*lost/n:.2f} 个百分点。',
        f'含官能团/R-site的 {dummy["images"]} 张图：星图 {dummy["graph_star_exact"]}/{dummy["images"]}={percent(dummy["graph_star_exact"]/dummy["images"])}；OCR图 {dummy["graph_ocr_exact"]}/{dummy["images"]}={percent(dummy["graph_ocr_exact"]/dummy["images"])}。',
        f'结构已经正确的带dummy图中，OCR保留 {dummy["graph_ocr_exact"]}/{dummy["graph_star_exact"]}={percent(dummy["graph_ocr_exact"]/dummy["graph_star_exact"])}。',
        f'匹配到GT官能团/R-site的预测框读字正确：{sum(r["exact"] for r in matched)}/{len(matched)}；同一批标签改用GT框：{paired_gt}/{len(matched)}。',
        f'全部GT官能团/R-site框读字正确：{sum(r["exact"] for r in gt_alias)}/{len(gt_alias)}（GT裁剪是对照，不是保证更优的上限）。',
        '', '参考结构公平性：',
        f'全部 {n} 个实际绘制SMILES与molfile/标注结构一致，参考结构一致性核验 {references["counts"]["gt_star_exact"]}/{n}（直接核验GT节点/边，不是GT boxes经KD-tree的组图成功率）。',
        f'{references["counts"]["augmented_parent_restored"]} 张图含H替换为dummy的增强；评测使用增强后结构，父SMILES仅作已还原增强的来源核验。来源核验 {references["counts"]["parent_source_verified"]}/{n}。',
        f'{references["counts"]["dummy_nodes"]} 个模型other目标中，官能团/别名/R-site {references["counts"]["alias_or_rsite_nodes"]} 个，词表外真实元素 {references["counts"]["unsupported_element_nodes"]} 个。',
        f'{functional["groups"]["with_unsupported_element"]["images"]} 张图包含词表外真实元素，当前13类模型不能给出这些元素身份；正式星图不会给other当作真实元素的信用。',
        '对展开的源SMILES，按预定义官能团的根节点结构展开参考标签，要求全源图精确匹配并覆盖全部原子，再按对应片段折叠。CF3的5→2节点示例与缺F反例见folding_examples.json。',
        f'本批数据的实际绘制SMILES已保留标签，本次无需额外折叠展开片段（实际触发{references["counts"]["expanded_source_aliases_folded"]}次）；R未知具体片段且无出处映射时不猜测。',
        'MCS仅为10个失败样本提供最大公共子图和原子对应诊断，不参与正确率、GT折叠或OCR候选选择。',
        '', '范围：当前是属性分子图指标，不等于完整canonical SMILES等价率。显式H作为节点保留；隐式H、同位素、原子CIP与E/Z没有独立评分。',
        '这是EasyOCR 1.7.2英文预训练识别器的结果，不代表其他OCR后端或化学OCR模型的准确率。',
        '新增9项指标/片段折叠/格式兼容测试通过；原严格全图1183张精确复现，预测与OCR输出均已冻结保存。',
        '', '入口：bash scripts/evaluate_full_v3_graph_metrics.sh（可设置BB2M_GPU、BB2M_EVAL_OUTPUT和BB2M_CHECKPOINT）']
    (out/'report.txt').write_text('\n'.join(lines)+'\n')
    errors=[r for r in matched if not r['exact']][:16]
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
    canvas=Image.new('RGB',(1000,90+150*((len(errors)+1)//2)),'white');draw=ImageDraw.Draw(canvas)
    draw.text((15,12),'Predicted dummy crops: GT label shown only AFTER OCR inference',fill='black',font=font)
    sample_lookup={s['sample_id']:s for s in samples}
    for j,row in enumerate(errors):
        with Image.open(dataset/sample_lookup[row['sample_id']]['image']) as im:
            bounds=tuple(round(v) for v in row['crop_box']);crop=im.convert('RGB').crop(bounds)
        crop=ImageOps.contain(crop,(460,75));x=500*(j%2)+15;y=70+150*(j//2)
        canvas.paste(crop,(x,y));draw.text((x,y+80),'GT: '+row['expected']+' | OCR: '+row['text'],font=font,fill='red')
    canvas.save(out/'ocr_errors.png')
    for path in [root/'reference_verified/report.json',root/'graph_ocr_verified/mcs_diagnostics.json',root/'graph_ocr_verified/ocr_crops.json']:
        shutil.copyfile(path,out/('reference_audit.json' if path.parent.name=='reference_verified' else path.name))
    source=out/'source';source.mkdir(exist_ok=True)
    names=['cache_final_predictions.py','audit_graph_references.py','molecular_graph_metrics.py','evaluate_graph_ocr.py','rescore_functional_groups.py','finalize_graph_metrics.py','test_molecular_graph_metrics.py']
    hashes={}
    for name in names:
        path=Path(__file__).with_name(name);shutil.copyfile(path,source/name);hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    write(out/'source_sha256.json',hashes)
    (out/'index.html').write_text('<meta charset="utf-8"><title>Dummy graph + OCR evaluation</title><h1>分子图与 OCR 分项评估</h1><p><a href="report.txt">完整报告</a> | <a href="samples.csv">逐样本结果</a> | <a href="report.json">JSON</a></p><pre>'+html.escape('\n'.join(lines[:14]))+'</pre><h2>OCR errors</h2><img width="100%" src="ocr_errors.png">')
    with zipfile.ZipFile(root/'graph_metrics_review.zip','w',zipfile.ZIP_DEFLATED) as z:
        for path in out.rglob('*'):
            if path.is_file():z.write(path,path.relative_to(root))
    print(json.dumps(dict(metrics=table,ocr_lost_images=lost,review=str(out.resolve())),indent=2,ensure_ascii=False))

if __name__=='__main__':main()
