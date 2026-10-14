"""Target-specific reports saved beside each new run; historical reports stay intact."""
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', '/tmp/oxygen-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve
from . import config
from .evaluation import metrics, alarm_statistics


def table(frame, columns):
    def fmt(x):
        if isinstance(x, (float, np.floating)):
            return f'{x:.4f}' if np.isfinite(x) else 'undefined'
        return str(x)
    return '| ' + ' | '.join(columns) + ' |\n| ' + ' | '.join(['---'] * len(columns)) + ' |\n' + '\n'.join(
        '| ' + ' | '.join(fmt(row[c]) for c in columns) + ' |' for _, row in frame.iterrows())


def evaluate_events(predictions, events, target):
    """Opportunity-based event sensitivity, with early censored onsets excluded.

    Every counted event has a full raw 30s lead-in and at least one eligible held-out
    prediction in [onset-horizon,onset). Not every eligible state must span all 30s:
    rows during earlier lows are ineligible for the onset target. Opportunity count
    and observed pre-onset coverage are explicitly saved. Alerts may match nearby
    events, so this does not estimate one-to-one alarm/event precision.
    """
    rows = []
    for name, group in predictions.groupby('Model', sort=False):
        for event in events.to_dict('records'):
            r = event['Record'];onset = event['Onset_s']
            numeric = pd.read_csv(config.DATA_DIR / f'bidmc_{int(r):02d}_Numerics.csv')
            numeric.columns = numeric.columns.str.strip()
            nt = numeric['Time [s]'].to_numpy(float);sv = numeric.SpO2.to_numpy(float)
            prior = (nt >= onset - config.HORIZON) & (nt <= onset)
            valid = np.isfinite(sv[prior]) & (sv[prior] > 0) & (sv[prior] <= 100)
            status = 'eligible'
            if not event['Qualifying_Onset']:
                status = 'left_censored_or_unknown_prior'
            elif onset - config.HORIZON < nt[0] or not valid.all() or np.diff(np.r_[onset - config.HORIZON, nt[prior], onset]).max(initial=0) > config.MAX_NUMERIC_GAP:
                status = 'incomplete_raw_lead_in'
            q = group[(group.Record == r) & (group.time_s >= onset - config.HORIZON) & (group.time_s < onset)]
            if target == 'future_low':
                q = q[q.spo2 > config.SPO2_THRESHOLD]
            if status == 'eligible' and q.empty:
                status = 'no_eligible_prediction_opportunity'
            alarms = q[q.Prediction == 1]
            rows.append({**event, 'Model': name, 'Evaluation_Status': status, 'Opportunities': len(q),
                         'Available_Lead_Time_s':onset-q.time_s.min() if len(q) else np.nan,
                         'Detected':bool(len(alarms)) if status == 'eligible' else np.nan,
                         'Lead_Time_s':onset-alarms.time_s.min() if status == 'eligible' and len(alarms) else np.nan})
    matched = pd.DataFrame(rows)
    summaries = []
    for name, g in matched.groupby('Model', sort=False):
        for subset in ['all_qualifying_onsets','91_92_episode_subset']:
            selected = g if subset == 'all_qualifying_onsets' else g[g.Has_91_92]
            eligible = selected[selected.Evaluation_Status == 'eligible']
            summaries.append({'Model':name,'Subset':subset,'Eligible_Events':len(eligible),
                              'Detected':int(eligible.Detected.eq(True).sum()),
                              'Sensitivity':eligible.Detected.eq(True).mean() if len(eligible) else np.nan,
                              'Median_Lead_Time_s':eligible.Lead_Time_s.median(),
                              'Excluded_Events':len(selected)-len(eligible),
                              'Positive_Recordings':eligible.Record.nunique()})
    return matched, pd.DataFrame(summaries)


def generate_target_report(output):
    output = Path(output)
    manifest = json.loads((output / 'run_manifest.json').read_text())
    target = manifest['target_definition']['name']
    predictions = pd.read_csv(output / 'per_record_predictions.csv')
    comparison = pd.read_csv(output / 'model_comparison.csv')
    recording = pd.read_csv(output / 'per_recording_metrics.csv')
    recording['All_Positive'] = recording.Positives.eq(recording.Samples)
    recording['Short_Eligible_Exposure'] = recording.Exposure_Hours.mul(3600).lt(config.HORIZON)
    recording['Recording_Class'] = np.where(recording.Positives > 0, 'positive_containing', 'negative_only')
    recording.to_csv(output / 'per_recording_metrics.csv', index=False)
    grouped = recording.groupby(['Model','Recording_Class']).agg(Recordings=('Record','size'), Macro_Recall=('Recall','mean'),
                   Macro_Precision=('Precision','mean'), Macro_Specificity=('Specificity','mean'), FP=('FP','sum'), FN=('FN','sum'))
    grouped.to_csv(output / 'recording_class_metrics.csv')
    recording[recording.Positives>0].to_csv(output/'positive_recording_detail.csv',index=False)
    matched, event_summary = evaluate_events(predictions,pd.read_csv(output / 'ground_truth_events.csv'),target)
    matched.to_csv(output / 'event_metrics.csv',index=False)
    event_summary.to_csv(output / 'event_summary.csv',index=False)
    stats=[];curves=[]
    fig,ax=plt.subplots(figsize=(7,5))
    for name,g in predictions.groupby('Model',sort=False):
        for label,part in g.groupby('Label'):
            stats.append({'Model':name,'Label':label,**part.Score.describe(percentiles=[.05,.5,.95]).to_dict()})
        precision,recall,threshold=precision_recall_curve(g.Label,g.Score)
        ax.plot(recall,precision,label=name)
        curves.extend({'Model':name,'Threshold':t,'Recall':r,'Precision':p} for t,r,p in zip(threshold,recall[:-1],precision[:-1]))
    pd.DataFrame(stats).to_csv(output/'score_distributions.csv',index=False)
    pd.DataFrame(curves).to_csv(output/'held_out_pr_curve.csv',index=False)
    ax.set(xlabel='Recall',ylabel='Precision',title=target+' — held-out pooled scores');ax.legend()
    fig.tight_layout();fig.savefig(output/'precision_recall.png',dpi=140);plt.close(fig)
    data=pd.read_csv(output/'causal_dataset.csv')
    references=[]
    for name,p in [('Always_Alert',np.ones(len(data),int)),('Never_Alert',np.zeros(len(data),int))]:
        references.append({'Model':name,**metrics(data.Label,p.astype(float),p)})
    pd.DataFrame(references).to_csv(output/'constant_references.csv',index=False)
    # Matched historical before/after comparisons. New-onset "before" is a relabelled,
    # frozen historical prediction set, NOT historical training on onset labels.
    old=pd.read_csv(config.OUTPUT_DIR/'per_record_predictions.csv')
    before=[];before_predictions=[]
    for name,g in old.groupby('Model',sort=False):
        aligned=g.merge(data[['Record','pulse_row','Label']],on=['Record','pulse_row'],suffixes=('_old',''),validate='one_to_one')
        before_predictions.append(aligned)
        alarm_parts=[alarm_statistics(part) for _,part in aligned.groupby('Record')]
        false=sum(a['False_Alert_Episodes'] for a in alarm_parts);exposure=sum(a['Exposure_Hours'] for a in alarm_parts)
        before.append({'False_Alert_Episodes':false,'Exposure_Hours':exposure,'False_Alerts_Per_Hour':false/exposure if exposure else np.nan,'Model':name,'Target':target,'Compared_Rows':len(aligned),
                       'Before_Interpretation':'frozen historical future-low model on target-eligible rows',
                       **metrics(aligned.Label,aligned.Score,aligned.Prediction)})
    pd.DataFrame(before).to_csv(output/'before_frozen_predictions.csv',index=False)
    before_events,before_summary=evaluate_events(pd.concat(before_predictions,ignore_index=True),pd.read_csv(output/'ground_truth_events.csv'),target)
    before_events.to_csv(output/'before_event_metrics.csv',index=False);before_summary.to_csv(output/'before_event_summary.csv',index=False)
    cols=['Model','Recall','Precision','Specificity','PR_AUC_AP','FP','FN','Macro_Recall','Macro_F1','False_Alerts_Per_Hour']
    text='# Target-specific held-out evaluation\n\n'+json.dumps(manifest['target_definition'],indent=2)+'\n\n'
    text+='Fixed models and hyperparameters; no model search or threshold adjustment after testing. Pooled timestamp metrics and Macro_* recording metrics are distinct. Undefined metrics remain NaN. AP means average precision. Pooled scores combine refits and are uncalibrated.\n\n'
    text+=table(comparison,cols)+'\n\n'+table(pd.DataFrame(before),['Model','Recall','Precision','Specificity','PR_AUC_AP','FP','FN'])+'\n\n'
    text+='The second table uses frozen historical future-low models on the SAME eligible rows, relabelled for this target. It is a reference, not a prior new-onset-trained experiment. Primary future-low includes ongoing lows; onset requires current observed normal saturation. Targets and populations are not interchangeable.\n\n'
    text+=table(event_summary,['Model','Subset','Eligible_Events','Detected','Sensitivity','Median_Lead_Time_s','Excluded_Events'])+'\n\n'
    text+='Events require one observed low reading; persistence is not presumed. Missing readings/gaps break episodes; left-censored onsets are excluded. A full valid raw lead-in plus at least one eligible prediction opportunity is required; opportunity coverage is saved. Alerts in [onset-horizon,onset) count. Nearby events can share an alert. The 91–92 subset refers to actual readings within the episode, not exact SpO2 prediction. Always-alert would detect every eligible event; event sensitivity must be read with FP burden.\n\n'
    text+='False alarm episodes use the existing definition: positive runs split by negative decisions or >configured alert gap, false if the run contains no positive target. Evaluable exposure sums adjacent eligible pulse intervals <=that gap. It excludes current-low periods for onset; it is not total recording time. A sustained alarm can yield few episode starts yet high false-positive duty. No suppression policy has been tuned.\n\n'
    text+='All-positive and short eligible exposures are explicitly flagged in positive_recording_detail.csv; they are retained, but do not establish within-record discrimination. Few independent positive recordings, unverified patient identity/monitor delay and one-second numeric precision limit inference. This is retrospective recording-wise CV, not an untouched external test set. Model parameters remain fixed from the earlier run.\n'
    if manifest.get('benchmark_completed'):
        latency=pd.read_csv(output/'latency_benchmark.csv')
        text+='\n'+table(latency,['Model','Component','Median_ms','P95_ms','P99_ms','Repetitions'])+'\n\n'
        text+='Single native thread; warmed perf_counter_ns timings. Model loading measured separately over 100 repetitions; P99 withheld below configured minimum. Pulse components use 2,000 measurements; waveform replay is per incoming sample and separately per confirmed pulse. Synthetic repeat measurements characterize the API path, not physiological distributions. Numeric lookup outside replay timers simulates already-available monitor input. Computational timings exclude acquiring signals, filter phase delay, communication and actuation. Pulse confirmation adds one 125Hz sample; 30 seconds is a label horizon, not guaranteed lead time.\n'
    (output/'EVALUATION.md').write_text(text)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    generate_target_report(parser.parse_args().output)
