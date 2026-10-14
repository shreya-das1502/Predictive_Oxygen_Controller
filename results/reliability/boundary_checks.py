from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from src import config
results=[]
for target in config.TARGETS:
 d=Path('results/final')/target;m=json.loads((d/'run_manifest.json').read_text())
 assert not any('spo2' in f.lower() or 'Label' in f for f in m['configuration']['FEATURES'])
 for split in m['recording_splits']:
  outer=set(split['outer_train_records']);assert split['test_record'] not in outer
  val=[]
  for inner in split['inner']:
   assert not set(inner['train_records'])&set(inner['validation_records'])
   assert set(inner['train_records'])|set(inner['validation_records'])==outer
   val.extend(inner['validation_records'])
  assert sorted(val)==sorted(outer)
 data=pd.read_csv(d/'causal_dataset.csv')
 oof=pd.read_csv(d/'validation_oof_predictions.csv.gz');held=pd.read_csv(d/'per_record_predictions.csv')
 assert (oof.Record!=oof.Outer_Test_Record).all()
 assert not oof.duplicated(['Model','Outer_Test_Record','sample_id']).any()
 count=0
 for (model,test),g in oof.groupby(['Model','Outer_Test_Record']):
  assert set(g.sample_id)==set(data.loc[data.Record.ne(test),'sample_id'])
  positive=np.sort(g.loc[g.Label.eq(1),'Score'].to_numpy())[::-1]
  threshold=positive[int(np.ceil(.92*len(positive)))-1]
  assert g.Threshold.nunique()==1 and np.isclose(threshold,g.Threshold.iloc[0])
  h=held[(held.Model==model)&(held.Record==test)]
  assert np.isclose(h.Threshold,threshold).all()
  np.testing.assert_array_equal((h.Score>=h.Threshold).astype(int),h.Prediction)
  count+=1
 results.append({'Target':target,'Outer_Model_Folds':count,'Training_OOF_Rows':len(oof),'Threshold_Recomputation':'passed','Split_Disjointness':'passed','No_SpO2_Feature':'passed'})
Path('results/reliability/boundary_checks.json').write_text(json.dumps(results,indent=2));print(results)
