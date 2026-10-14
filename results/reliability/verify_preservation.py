from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from src.check_project import verify_links
root=Path.cwd()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
preserved=json.loads(Path('results/reliability/preservation.json').read_text())
for p,want in preserved.items():assert sha(p)==want,p
manifest=json.loads(Path('results/reliability/final/manifest.json').read_text())
for p,want in manifest['inputs'].items():assert sha(p)==want,p
for p,want in manifest['outputs'].items():assert sha(Path('results/reliability/final')/p)==want,p
assert sha('src/reliability_evaluation.py')==manifest['code_sha256']
raw=json.loads(Path('results/final/future_low/run_manifest.json').read_text())['source_hashes']
for p,want in raw.items():assert sha(p)==want,p
for target in ['future_low','new_onset']:
 hashes=json.loads(Path('results/final',target,'run_manifest.json').read_text())['code_hashes']
 for name in ['config.py','dataset.py','features.py','preprocessing.py','labeling.py','model.py','controller.py','evaluation.py','provenance.py']:
  p=root/'src'/name;assert sha(p)==hashes[str(p)],str(p)
checks=pd.read_csv('results/reliability/final/independent_checks.csv');assert checks.Discrepancies.sum()==0
matches=pd.read_csv('results/reliability/final/window_event_matches.csv')
primary=matches[(matches.Method=='episode_start_one_to_one')&matches.Detected.eq(True)]
assert not primary.duplicated(['Target','Model','Window_s','Record','Alarm_ID']).any()
assert (primary.Lead_Time_s>0).all() and (primary.Lead_Time_s<=primary.Window_s).all()
for (_,window),g in matches.groupby(['Target','Window_s']):
 ref=None
 for _,m in g.groupby(['Model','Method']):
  values=m.sort_values(['Record','Event_ID'])[['Record','Event_ID','Evaluation_Status','Opportunities']].reset_index(drop=True)
  if ref is None:ref=values
  else:pd.testing.assert_frame_equal(ref,values)
log=Path('results/reliability/test_results.txt').read_text();assert 'Ran 91 tests' in log and log.rstrip().endswith('OK') and 'skipped=' not in log
links=verify_links()
q={'status':'passed','tests':{'count':91,'skipped':0,'failures':0,'errors':0,'seconds':11.330,'log_sha256':sha('results/reliability/test_results.txt')},
   'protected_existing_artifacts_unchanged':len(preserved),'raw_sources_unchanged':len(raw),'independent_calculation_checks':len(checks),'discrepancies':int(checks.Discrepancies.sum()),
   'boundary_checks':json.loads(Path('results/reliability/boundary_checks.json').read_text()),'identical_eligibility_across_all_models_baselines_and_methods':True,
   'one_to_one_matching_no_duplicate_episode_credit':True,'actual_saved_events_unchanged_after_boundary_correction':json.loads(Path('results/reliability/correction_verification.json').read_text()),
   'local_documentation_links':links,'predictor_label_feature_config_source_hashes_match_saved_runs':True,'final_manifest_inputs_outputs_and_code_verified':True,
   'audit_source_hashes':{str(p):sha(p) for p in [Path('src/reliability_evaluation.py'),Path('src/target_reporting.py'),Path('tests/test_reliability.py'),*Path('results/reliability').glob('*.py')]},
   'executed_commands':['python -m src.reliability_evaluation --output results/reliability/final','PYTHONPATH=. python results/reliability/boundary_checks.py','python results/reliability/shared_alert_check.py','PYTHONPATH=. python results/reliability/original_reference_check.py','python -m unittest discover -s tests -v','python -m src.check_project --links','git diff --check'],
   'no_model_training_or_threshold_changes':True}
Path('results/reliability/verification.json').write_text(json.dumps(q,indent=2))
print(json.dumps({k:v for k,v in q.items() if k in ['status','tests','protected_existing_artifacts_unchanged','raw_sources_unchanged','independent_calculation_checks','local_documentation_links']},indent=2))
