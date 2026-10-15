"""Audit existing evidence without regenerating any historical output."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src import config
from src.check_project import verify_run, audit_data
from src.evaluation import metrics
from src.alarm_budget_experiment import verify_original_inputs
from src.local_30_evaluation import verify_splits

ROOT = config.ROOT


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot():
    roots=('src','data','results','archive','docs','dashboard','signal_studies','tests')
    return {str(p.relative_to(ROOT)):sha(p) for base in roots for p in sorted((ROOT/base).rglob('*'))
            if p.is_file() and not p.is_symlink() and '__pycache__' not in p.parts}


def verify_snapshot(values):
    bad=[p for p,w in values.items() if not (ROOT/p).is_file() or sha(ROOT/p)!=w]
    if bad:raise ValueError('Protected file changed: '+str(bad))
    return len(values)


def audit(output):
    output=Path(output).resolve()
    if (ROOT/'results/ml_improvement').resolve() not in output.parents or output.exists():
        raise ValueError('Choose a new results/ml_improvement/<run> directory')
    protected=snapshot(); output.mkdir(parents=True)
    (output/'preservation.json').write_text(json.dumps(protected,indent=2))
    b=ROOT/'results/alarm_budget_experiment/run02'
    m=json.loads((b/'manifest.json').read_text())
    for section,folder in [('inputs',ROOT),('outputs',b)]:
        for p,w in m[section].items():
            if sha(folder/p)!=w:raise ValueError('Baseline manifest mismatch: '+p)
    saved=pd.read_csv(b/'outer_predictions.csv.gz',float_precision='round_trip')
    recomputed=[]; dataset=[]; checks=[]
    for target in config.TARGETS:
        directory=ROOT/'results/final'/target
        manifest=json.loads((directory/'run_manifest.json').read_text())
        verify_original_inputs(directory,manifest)
        checks.append(verify_run(directory))
        d=pd.read_csv(directory/'causal_dataset.csv');verify_splits(manifest['recording_splits'],d.Record.unique())
        dataset.append(dict(Target=target,Rows=len(d),Positives=int(d.Label.sum()),Recordings=d.Record.nunique(),Positive_Recordings=int(d.groupby('Record').Label.sum().gt(0).sum())))
        old=pd.read_csv(directory/'per_record_predictions.csv');comparison=pd.read_csv(directory/'model_comparison.csv').set_index('Model')
        for model,g in old.groupby('Model'):
            v=metrics(g.Label,g.Score,g.Prediction)
            for key in ['TP','FP','FN','TN','Recall','Precision','Specificity','PR_AUC_AP']:
                if not np.isclose(v[key],comparison.loc[model,key],equal_nan=True):raise ValueError('Historical metric mismatch')
            recomputed.append(dict(Stage='historical_recall',Target=target,Model=model,**v))
    for (target,model,budget),g in saved.groupby(['Target','Model','Budget']):
        np.testing.assert_array_equal(g.Prediction,(g.Score>=g.Threshold).astype(int))
        recomputed.append(dict(Stage='final_budget_common_normal',Target=target,Model=model,Budget=budget,**metrics(g.Label,g.Score,g.Prediction)))
    summary=pd.read_csv(b/'summary.csv');per=pd.read_csv(b/'per_recording.csv');events=pd.read_csv(b/'event_matches.csv')
    for row in summary.itertuples():
        p=per[(per.Target==row.Target)&(per.Model==row.Model)&(per.Budget==row.Budget)&(per.Window_s==row.Window_s)&(per.Subset==row.Subset)]
        for key in ['Detected','Eligible_Events','Missed_Events','TP','FP','FN','TN','Exposure_s','Alarm_Active_s','Off_Event_Alarm_Starts']:
            if not np.isclose(p[key].sum(),getattr(row,key)):raise ValueError('Baseline aggregation mismatch: '+key)
        e=events[(events.Target==row.Target)&(events.Model==row.Model)&(events.Budget==row.Budget)&(events.Window_s==row.Window_s)&events.Evaluation_Status.eq('eligible')]
        if row.Subset=='91_92':e=e[e.Has_91_92]
        if len(e)!=row.Eligible_Events or e.Detected.eq(True).sum()!=row.Detected:raise ValueError('Event denominator mismatch')
    selected=pd.read_csv(b/'selected_thresholds.csv')
    training=selected.groupby(['Target','Model']).agg(Selections=('Detected','size'),Training_Detections=('Detected','sum'),Silent_Selections=('No_Training_Alerts','sum')).reset_index()
    pd.DataFrame(recomputed).to_csv(output/'baseline_timestamp_metrics.csv',index=False)
    pd.DataFrame(dataset).to_csv(output/'baseline_dataset.csv',index=False)
    summary.to_csv(output/'baseline_event_summary.csv',index=False)
    training.to_csv(output/'baseline_selection_diagnostics.csv',index=False)
    (output/'baseline_checks.json').write_text(json.dumps(dict(status='verified',quality_gates=checks,raw=audit_data(),protected_count=len(protected),manifest_inputs=len(m['inputs']),manifest_outputs=len(m['outputs'])),indent=2))
    (output/'baseline_complete.json').write_text(json.dumps({p.name:sha(p) for p in output.glob('baseline_*')},indent=2))
    verify_snapshot(protected)
    report=ROOT/'docs/ML_IMPROVEMENT_BASELINE.md'
    text='''# Verified ML improvement baseline\n\nThe existing 30 local recordings are the entire cohort. No raw dataset, trained model, saved output, historical report or integrity receipt was changed. This audit was completed before fitting experimental models.\n\n## Verified design\n\n`src/config.py:MODEL_CONFIGS` defines ELM (50 tanh nodes, ridge 1, positive weight 5, seed 42), balanced logistic regression (C=1), GaussianNB (equal priors, smoothing 1e-6) and a balanced depth-3 tree (leaf minimum 50). `src/run_experiments.py`, `src/evaluation.py:grouped_oof` and `src/model.py:FittedModel.fit` implement outer leave-one-recording-out and inner three-fold recording-grouped OOF predictions; imputation/scaling fit inside each training fold. Thresholds receive only outer-training OOF predictions. Verified recording separation does not establish patient independence: source headers and inventories provide no verified patient identifiers.\n\n`src/labeling.py` defines future-low as minimum SpO2 <=94 over (t,t+30], complete valid coverage, maximum gap 1.01 s; ongoing lows count. New-onset additionally requires fresh current SpO2 >94 and an adjacent valid >94 to <=94 crossing. One reading suffices; no clinical persistence requirement. Current SpO2 gates eligibility and supplies labels; it is not a predictor. `src/preprocessing.py` and `src/features.py` provide forward-only filtering, next-sample pulse confirmation, and current/past pulse features. No PTT is implemented.\n\n**Horizon correction:** all existing models were trained for 30 seconds. The reported 5/10/15/30 seconds in event tables are look-back matching windows, not four trained forecast horizons. Strict current-normal event exposure is 10,329.032 s (2.869176 h); 15 events qualify at 5/10/15 s, 14 at 30 s, and three are the observed-91/92 subset. One-to-one chronological earliest unused alarm starts in [onset-W,onset) receive credit. Off-event starts have no qualifying onset in (start,start+W]; active fraction uses left-held duration, not alerted pulse fraction.\n\n## Reproduced data and timestamp results\n\n'''
    text+=pd.DataFrame(dataset).to_string(index=False)+'\n\n```text\n'+pd.DataFrame(recomputed).query("Stage == 'historical_recall'")[['Target','Model','Recall','Precision','Specificity','PR_AUC_AP','TP','FP','FN','TN']].to_string(index=False)+'\n```\n'
    text+='''\nThe full future-low population has 20,157 rows/4,587 positives/30 recordings; onset has 16,050/501/26 nonempty groups, with six positive-containing recordings. Four all-low recordings have zero new-onset opportunities, not missing negative rows. Overlapping future windows create many correlated positive pulse labels per event.\n\n## Reproduced final operating-policy result\n\nAll four learned models have **0/14** eligible 30 s detections, **0/15** at 5/10/15 s and **0/3** severe-interest detections, for both training-target families and budgets 1/5/10 starts/evaluable hour with 5% active cap. All matched learned leads are undefined, not zero. These budgets are research assumptions, not clinically validated limits. Event rows/counts and timestamp counts were recomputed from original saved arrays, and original manifest input/output SHA-256 values and both saved-run quality gates passed. New machine-readable copies/recomputations are in the isolated output directory.\n\n## Failure analysis: evidence versus hypotheses\n\n**Verified:** earlier onset ELM AP is 0.023976 against 501/16050=0.031215 prevalence; logistic AP 0.033500, GaussianNB 0.057595 and tree 0.039582. Poor discrimination precedes threshold choice; high future-low recall partly measures ongoing lows. The scalar-cutoff policy imposes both start and duration constraints, so frequent or sustained alerts can be infeasible. Training-selected policies are often silent; the table below counts them. Nonzero training detections did occur, so zero outer detection is not explained by universally zero training sensitivity. Some outer-record constraints fail despite training feasibility. Scores from inner refits and outer refits can have different distributions. No verified evaluation defect was found in this audit.\n\n**Hypotheses to test, not established causes:** short/low-dimensional pulse features may not capture pre-onset physiology; causal longer context may help, but waveform/peak quality is not clinically validated. Inner-to-outer score-scale transfer may impair raw cutoffs; an empirical training-negative score rank may improve scale comparability, but is not probability calibration. Shorter truly trained horizons may change discrimination; existing shorter look-back counts do not answer that question. Few independent event-bearing recordings, correlated samples, unknown monitor delay and missing readings fundamentally limit inference. A high-capacity family or broad search is not justified for this pass.\n\n```text\n'''+training.to_string(index=False)+'''\n```\n\n## Commands and preservation\n\n`OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python -m ml_research.baseline --output results/ml_improvement/run01` verifies the manifests, both original target runs, event aggregation and timestamp results and creates only new files. The output path must not already exist. Explicit saved-run checks: `python -m src.check_project --verify-run results/final/new_onset` and the corresponding `future_low` command.\n\nThree pre-existing full-suite failures concern authorized historical document changes: `test_target_hardening.SavedTargetTests.test_historical_outputs_preserved`, `test_local_30.Local30ArtifactTests.test_outputs_and_protected_files_are_unchanged`, and `test_alarm_budget.AlarmBudgetArtifactTests.test_inputs_outputs_and_preservation_verified`. Their receipts/assertions remain protected. Their old hashes are accurate historical baselines; this task will not rewrite them to obtain a green suite.\n\nThe cohort has already been examined repeatedly. Any ensuing controlled local experiment is exploratory held-out-recording evaluation, not a fresh confirmatory trial, unseen-patient validation or evidence for oxygen actuation. No improvement is promised.\n'''
    (output/'ML_IMPROVEMENT_BASELINE.md').write_text(text)
    if not report.exists():report.write_text(text)
    print('Baseline verified:',output,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);audit(p.parse_args().output)
