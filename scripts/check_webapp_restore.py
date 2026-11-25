#!/usr/bin/env python3
"""Exercise review/revision persistence in disposable storage, never user annotations."""
import copy,hashlib,json,os,subprocess,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory(prefix='bb2m-webapp-test-') as directory:
 data=Path(directory)
 subprocess.run([sys.executable,str(ROOT/'scripts/restore_webapp_examples.py'),'--storage',str(data)],check=True,stdout=subprocess.DEVNULL)
 for key,relative in {'I2BL_RESULTS_DATA_DIR':'results_data','I2BL_FIXED_DATA_DIR':'fixed_data',
                     'I2BL_TRAINING_PAIRS_DIR':'fixed_data/training_pairs','I2BL_EXAMPLES_DIR':'examples',
                     'I2BL_PASTED_EXAMPLES_DIR':'pasted_examples','I2BL_CLASSES':'classes.json'}.items():
  os.environ[key]=str(data/relative)
 os.environ['I2BL_INFERENCE_PROFILE']='legacy'
 os.environ['I2BL_KETCHER_DIR']=str(ROOT/'repo/i2bl_webapp/ketcher')
 sys.path.insert(0,str(ROOT/'repo/i2bl_webapp'))
 import webapp
 from fastapi.testclient import TestClient
 with TestClient(webapp.app) as client:
  assert client.get('/').status_code==200
  assert client.get('/api/config').json()['has_ketcher']
  assert client.get('/ketcher/index.html').status_code==200
  examples=client.get('/api/examples').json()['examples'];assert len(examples)==5
  for ex in examples:
   result=client.post('/api/open-example',data={'filename':ex['name']})
   assert result.status_code==200,result.text
   payload=result.json()
   assert len(payload['annotation_items']['b'])==ex['counts']['bonds']
   assert payload['annotation_revision']==0
   assert payload['annotation_review_status']=='imported_unreviewed'
  name='HF_2735c7e3f9cf_native.png'
  seed=data/'examples'/name.replace('.png','.annotation.json');original=seed.read_bytes()
  payload=client.post('/api/open-example',data={'filename':name}).json()
  sha=payload['image_sha256']
  assert sha==hashlib.sha256((data/'examples'/name).read_bytes()).hexdigest()
  for revision,label in [(1,'Cl'),(2,'R.')]:
   ac=copy.deepcopy(payload['annotation_items']['ac']);ac[0]['label']=label
   form={'result_stem':payload['result_stem'],'fix_mode':'detections',
         'ac_json':json.dumps({'image_size':payload['image_size'],'items':ac}),
         'b_json':json.dumps({'image_size':payload['image_size'],'items':payload['annotation_items']['b']}),
         'ocr_json':json.dumps(payload['ocr'])}
   result=client.post('/api/fix',data=form)
   assert result.status_code==200,result.text
   assert result.json()['training_pair']['revision']==revision
   latest=json.loads((data/'fixed_data/training_pairs'/sha/'ground_truth.json').read_text())
   assert latest['revision']==revision and latest['review_status']=='human_confirmed'
   assert latest['atoms'][0]['label']==label
   assert latest['saved_at_utc'] and latest['provenance']['source']=='transfer_training_snapshot'
   payload=client.post('/api/open-example',data={'filename':name}).json()
   assert payload['annotation_items']['ac'][0]['label']==label
   assert payload['annotation_revision']==revision and payload['annotation_review_status']=='reviewed'
   history=client.post('/api/load-history',data={'result_stem':payload['result_stem']}).json()
   assert history['found'] and history['revision']==revision
   assert len(client.get('/api/examples').json()['examples'])==5,'Fixed aliases duplicated original examples'
  revisions=list((data/'fixed_data/training_pairs'/sha/'revisions').glob('revision_*.json'))
  assert len(revisions)==2
  assert seed.read_bytes()==original,'Imported snapshot was overwritten'
  print('PASS: five examples; exact labels; Ketcher assets; edit/save r1/r2; reload latest; history; immutable revisions; no duplicate examples.')
print('All save tests used temporary storage; production examples remain unconfirmed.')
