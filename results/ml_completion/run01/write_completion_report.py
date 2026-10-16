"""Generate the completion report solely from already-saved measured tables."""
import argparse,json,os
from pathlib import Path
import pandas as pd
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from ml_research.baseline import ROOT
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve()
local=out/'ML_IMPROVEMENT_REPORT.md';global_doc=ROOT/'docs/ML_IMPROVEMENT_REPORT.md'
if local.exists() or global_doc.exists():raise ValueError('Inspect existing report; do not overwrite')
s=pd.read_csv(out/'event_summary.csv');c=pd.read_csv(out/'before_after_comparison.csv');ci=pd.read_csv(out/'recording_bootstrap_intervals.csv');ts=pd.read_csv(out/'timestamp_summary.csv');lat=pd.read_csv(out/'latency.csv');counts=pd.read_csv(out/'dataset_counts.csv')
q=s[s.FeatureSet.isin(['base10','context18'])&s.Subset.eq('all')&s.Window_s.eq(s.Horizon_s)]
hits=q[q.Detected.gt(0)];pooled=hits[hits.False_Alarm_Starts_Per_Hour.le(hits.Budget+1e-12)&hits.Alarm_Active_Fraction.le(.05+1e-12)]
r=c[c.Subset.eq('all')&c.Window_s.eq(c.Horizon_s)&c.After_Detected.gt(0)];strict=r[r.All_Exposed_Records_Within_Budget&r.All_Exposed_Records_Within_Active_Cap&r.After_False_Alarm_Starts_Per_Hour.le(r.Budget+1e-12)&r.After_Alarm_Active_Fraction.le(.05+1e-12)]
assert len(pooled)==3 and len(strict)==0
prefix=str(Path(os.path.relpath(out,ROOT/'docs')))+'/'
def link(n):return f'[{n}]({prefix+n})'
def table(frame):return '\n```text\n'+frame.to_string(index=False,float_format=lambda v:f'{v:.6g}')+'\n```\n'
text='''# Completed bounded ML follow-up

## Executive finding

The frozen follow-up is numerically complete: **three prespecified conditions improved pooled event detection from zero to one while meeting their pooled held-out alarm assumptions; zero detecting conditions met those assumptions on every exposed recording.** All three detections are the SAME record07 onset, not three independent successes. No useful general early-warning policy or overall model superiority is established. No model/horizon/budget was selected using these outer results. No clinical readiness, oxygen recommendation or validated feedforward/PID control is claimed.

The preserved original final policy remains **0/14 at W30, 0/15 at W5/10/15 and0/3 severe-interest**, for both original targets, all four families and all budgets. Original30s/base10/raw follow-up policies reproduced all192 baseline event rows before accepting the new summaries. This is a retrospective exploratory analysis of only30 repeatedly examined local recordings; patient identities are unverified.

## What was finished, rather than restarted

All1120 model/fold groups,6720 training-selected cutoffs and144 evaluation conditions already existed. This completion reuses their immutable scores, thresholds, labels,103680 per-record event rows and event matches. It runs **zero scientific CV/OOF refits and zero new policy searches**. Forty fixed outer-training-only refits solely restored missing objects for timing and reproduce each corresponding saved held-out score. They do not replace scientific predictions. Older source/outputs remain separate. The previously slow bootstrap now shares identical contribution calculations and computes the exact repeated-observation median with vectorized weighted order statistics; tests match the existing reference implementation exactly, including empty and tied leads.

One verified rendering defect was fixed outside frozen code: categorical all-missing leads caused matplotlib ConversionError. The corrected event panel uses numeric positions and omits undefined leads with an explicit no-detections annotation. The four already-created figures were preserved. No target, feature definition, preprocessing, estimator, threshold or event rule was changed to obtain detections.

## Frozen methods and causal boundaries

The original protocol and configuration are linked below. Four families retain the original parameters: ELM50 tanh units/ridge1/positive-weight5/seed42; balanced logistic C1/max_iter2000/random42; GaussianNB smoothing1e-6/equal priors; balanced depth3 tree/minleaf50/random42. Five settings: base10 features with actual label horizons H5/10/15/30, plus one context18 ablation at H30. Both future_low and new_onset are evaluated, with raw scores and training-negative empirical ranks. No extra classifier, feature search, rebalancing sweep or calibration search was added.

Future-low means minimum valid observed SpO2<=94 in(t,t+H], including ongoing lows. New-onset additionally requires fresh currentSpO2>94 and a valid adjacent >94 to<=94 crossing. Complete valid horizons, maximum1.01s numeric gaps and one-reading events retain their original definitions. The severe-interest subset contains observed91/92 readings, not a separately trained severe target. Current/futureSpO2, labels and IDs never enter predictor features. Forward filtering, pulse confirmation at availability time and past-only history remain unchanged. Eight context features use current/past amplitude/rate/interval statistics, trend, pulse count and prior interval; they are not validated SQI or PTT.

Outer evaluation is leave-one-recording-out. Inner three-fold GroupKFold, imputation, scaling, estimator fitting and negative-rank references use permitted training records only. The negative empirical CDF uses NEGATIVE FIT-row scores for each respective inner/outer fit; it is not probability calibration and may inherit training-score optimism/ties. A fixed training-OOF grid chooses maximum one-to-one event detections at the trained horizon, subject to research budgets1/5/10 starts per evaluable hour and5% active cap, with unchanged deterministic burden/activity/cutoff ties. These are research assumptions, not clinical standards. No outer outcomes choose a threshold/model.

All horizons retain the original complete30s sample population, intentionally excluding additional short-H tail opportunities. The four all-low recordings have zero onset exposure and remain in the30-record reporting/bootstrap roster. Population counts:
'''+table(counts)+'''
H is an actual label/training horizon. W5/10/15/30 is an event look-back window, not minimum guaranteed warning. Primary follow-up comparisons use W=H. Frozen30 controls remain trainedH30 even when evaluated with shorter labelH; fixed baselines have trainedH0. Changing shorter trainedH also changes selectionW; it is not an isolated timing-only intervention.

## Event and alarm results

All compared conditions use the same eligible event set for a given W:15 atW5/10/15,14 atW30, three severe-interest events. Common exposure is10329.032s=2.869176h. Alarm starts use chronological earliest-unused matching in[onset-W,onset), one credit per episode and event, no clipped restart or shared credit. Off-event starts have no qualifying observed onset in(start,start+W], including sensitivity-excluded real onsets. Active time is left-held positive duration/exposure. FP timestamps, legacy false episodes and off-event starts are different quantities. Lead=onset minus actual matched alarm start; misses have undefined leads.

Every detecting PRIMARY learned condition (all other primary learned conditions detect zero):
'''+table(hits[['Target','Horizon_s','FeatureSet','ScoreSpace','Model','Budget','Detected','Eligible_Events','Missed_Events','Sensitivity','Off_Event_Alarm_Starts','False_Alarm_Starts_Per_Hour','Alarm_Active_Fraction','Lead_Median_s']])+'''
Active fractions in the table are fractions, not percentages: H10=0.0377198%, H15=0.0244747%, H30=0.0504985% approximately. The matched onset is record07 at367s, an isolated94% reading; alarm starts357.176s forH10/H15 and353.560s forH30 yield9.824/9.824/13.440s. This is not a91/92 event. All three severe-interest events remain missed in the primary learned comparisons. ELM, logistic, tree, all future-low settings and the context18 ablation provide no primary event-detection gain.

These three pooled-feasible conditions use configured budget10 even though their observed pooled rates happen to be smaller; they are not post-hoc budget1/5 policies. Each violates the configured start budget on individual recordings: record26=10.154117/h forH10/H15; record7=16.125853/h and record26=10.154117/h forH30. Active cap passes for these three. Individual-record checks are a robustness analysis, not a replacement of the original pooled training objective. No robust model/policy is recommended from these outer tables.

Always/never-alert, periodic1s-in10s and pulse-rate>100 controls are saved for every target/labelH/budget/W. Always-alert's apparent event detections arise partly from eligibility-gap restarts and100% activity; it is not useful prediction or a compliant comparator. Never-alert's zero detections/burden is a negative reference, not a clinical strategy.

## Before/after evidence, timestamp metrics and feature analysis

'''+link('before_after_comparison.csv')+' contains every before/after event comparison; '+link('paired_recording_bootstrap.csv')+''' contains paired recording-resampled changes. Frozen original controls have zero detections, and the three point gains above are1/15,1/15,1/14. Complete budgets/windows/subsets/conditions are retained, not just favorable rows.

The following prespecified descriptive slice is H30/base10/raw/budget5, FULL target pooled timestamp classification; it is not event sensitivity or a selected winner:
'''+table(ts[ts.Horizon_s.eq(30)&ts.FeatureSet.eq('base10')&ts.ScoreSpace.eq('raw')&ts.Budget.eq(5)&ts.Metric_Scope.eq('full_target')][['Target','Model','Recall','Precision','Specificity','PR_AUC_AP','FP','FN','Macro_PR_AUC_AP']])+'''
Full timestamp tables separately label full_target versus current_normal populations, pooled versus defined-record macro metrics, and positive-containing versus negative-only recordings. AP is average precision, not trapezoidal area. Ongoing-low future-low performance cannot be substituted for onset warning. Training-only feature means/standardized differences are descriptive, not feature selection; no held-out feature importance drove the ablation. The original onset ELM AP0.023976 versus prevalence501/16050=0.031215 demonstrates weak ranking before threshold choice; it does not prove a specific physiological failure mechanism.

## Uncertainty, concentration and limitations

There are2000 shared whole-record bootstrap draws, seed20261015. Each recording contributes its complete counts/exposure/leads; no rematching across record copies. For all three detecting primary conditions sensitivity95% percentile intervals are[0,0.428571], with1996 defined event-bearing replicates. These wide intervals include zero. Lead distributions condition on detection of one original event; a degenerate resampled single-event lead is not precise population evidence. Zero-detection empirical[0,0] intervals do not establish true population sensitivity zero. Per-record influence removes only saved contributions for description; no difficult recording is removed from evaluation.

Bootstrap is conditional on saved overlapping CV fits: it does not refit models, establish patient independence, correct multiple comparisons, or create a fresh confirmatory cohort. Only six onset-bearing recordings,14/15 eligible onsets and three severe-interest events are available. Missing numerics, unvalidated beat/SQI extraction, correlated pulse windows, uncertain monitor delay and repeatedly examined outcomes constrain interpretation. No new data, patient IDs or synthetic clinical evidence were introduced.

Recovery provenance remains incomplete:1074 completed groups lost their in-memory OOF/reference hashes after interruption. Saved training grids/unlabeled scores and reconstructed deterministic splits remain; the missing execution hashes were not invented. The other46 groups have actual fit hashes. Frozen source, preserved input fingerprints, source row identities, original-policy parity, independent cutoff audit and new numerical checks supplement but do not erase this limitation. Per-fit wall-time history for original interrupted CV is unavailable; no total training-runtime claim is fabricated.

## Actual measured computation

The new timing table uses40 timing-only outer-training fits, with all40 last-record raw scores checked against saved scientific scores (rtol1e-9/atol1e-10). Warm batch-one timers include imputation/scaling, scoring, optional negative rank and the actual budget5 cutoff; causal pulse-feature calculation is timed separately on record01 pulse data. Each component has1000 measured repetitions after50 warmups. Median/P95/P99 values and raw samples are saved. The following is the H30/base10 slice plus feature timings; all other horizons/context variants are retained in the fullCSV:
'''+table(lat[(lat.FeatureSet.eq('base10')&lat.Horizon_s.eq(30))|lat.Component.eq('causal_pulse_features')][['Target','Horizon_s','FeatureSet','Model','ScoreSpace','Component','Repetitions','Median_ms','P95_ms','P99_ms']])+'''
This combined score/threshold timing is NOT model-only latency. It excludes raw-waveform acquisition/filtering, model loading and physiological sampling/monitor delay. The historical benchmark in results/final/<target>/latency_benchmark.csv separately measures those computational components; it is not a new context18 end-to-end benchmark. Do not add component percentiles to claim end-to-endP95 or turn the30s horizon into achieved warning. No new model superiority claim follows from timing.

## Reproducible artifacts and commands

'''+ '\n'.join('- '+link(n) for n in ['protocol.json','cloned_inputs.json','dataset_counts.csv','selected_thresholds.csv','fold_ledger.json','outer_predictions.csv.gz','timestamp_summary.csv','timestamp_per_recording.csv','event_summary.csv','event_per_recording.csv','event_matches.csv','recording_bootstrap_intervals.csv','recording_influence.csv','training_feature_analysis.csv','training_policy_summary.csv','latency.csv','latency_samples.csv.gz','timing_model_score_parity.csv','aggregation_verified.json'])+'''

These are continuation commands on already prepared input copies; stages reject completed outputs. For a separate reproduction choose a fresh directory instead of overwriting run01. Original fitting reproduction remains in the preserved frozen experiment scripts; do not rerun it to finish these artifacts.

```bash
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python -m ml_research.finalize_saved --output results/ml_completion/run01 --stage aggregate
MPLCONFIGDIR=/tmp/ml-completion-mpl OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python -m ml_research.finalize_saved --output results/ml_completion/run01 --stage timing
MPLCONFIGDIR=/tmp/ml-completion-mpl OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python -m ml_research.finalize_saved --output results/ml_completion/run01 --stage figures
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python results/ml_completion/run01/write_completion_report.py --output results/ml_completion/run01
```

Figures and their exact data sources:

'''+ '\n'.join('- '+link('figures/'+n)+': '+desc for n,desc in [('future_low_alarm_tradeoffs.png','Event sensitivity versus actual off-event starts/hour and active%; frozen controls and fixed baselines included.'),('new_onset_alarm_tradeoffs.png','Same trade-offs for onset; labels show true labelH and matchingW.'),('future_low_precision_recall.png','Pooled full-target timestamp precision/recall for each true labelH; prevalence reference.'),('new_onset_precision_recall.png','Same timestamp ranking comparison for onset; not calibrated probabilities.'),('event_counts_and_leads.png','All learned/frozen primary budget5 conditions; no matched leads, explicitly undefined, not zeros.')])+'''

The saved figures use the frozen renderer except the corrected all-missing-lead panel. File/source fingerprints and final verification are recorded in the run receipts. Existing dashboard playback continues to use original results/alarm_budget_experiment/run02, not these timing objects or a post-hoc selected policy. Its full-data demo remains demonstration-only. Updated submission text must retain that distinction.

## Recommendation and relationship to control

Some pooled point results improve, but a single nonsevere event with per-record constraint violations does not demonstrate useful robust early warning. Do not select an overall winner, preferred horizon or new cutoff from the outer table. A smallest next methodological study would compare ONE prespecified candidate and the frozen baseline with middle recording-grouped model/policy assessment inside each outer training split, preserving inner training-only calibration/cutoffs; this remains exploratory on the already examined30 recordings. No new experiment was run to pursue that recommendation.

This evidence can describe an offline risk signal's limitations. It provides no risk-to-dose law, oxygen actuator/plant, closed-loop stability, patient benefit or validated P/PI/PID/feedforward superiority. No oxygen-delivery controller was implemented.

Tests, independent saved-array verification, documentation preservation and readiness are recorded in ML_COMPLETION_TRACKER.md and the completion verification receipts. Historical hash failures must remain disclosed; no whole-suite-green or browser-visual claim is made.
'''
local.write_text(text.replace(prefix,''));global_doc.write_text(text)
(out/'scientific_conclusion.json').write_text(json.dumps(dict(pooled_feasible_detecting_conditions=len(pooled),all_record_feasible_detecting_conditions=len(strict),independent_detected_primary_events=1,model_selected=False,clinical_validation=False),indent=2))
print('Report created:',global_doc)
