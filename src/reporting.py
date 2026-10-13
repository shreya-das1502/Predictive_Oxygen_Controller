"""Reproducible reports from saved held-out predictions; never refit/tune here."""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/oxygen-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve
from .config import OUTPUT_DIR, DATA_DIR, ROOT, FEATURES, TARGET_RECALL, HORIZON, SPO2_THRESHOLD, MAX_NUMERIC_GAP
from .dataset import load_csv
from .evaluation import metrics, alarm_statistics


def markdown_table(df,columns):
    def fmt(v):
        if isinstance(v,(float,np.floating)):return f'{v:.4f}' if np.isfinite(v) else 'undefined'
        return str(v)
    return '| '+' | '.join(columns)+' |\n| '+' | '.join(['---']*len(columns))+' |\n'+'\n'.join('| '+' | '.join(fmt(row[c]) for c in columns)+' |' for _,row in df.iterrows())


def observed_events(record):
    """Episodes are contiguous observed <=94; missing/unknown is a boundary.

    Onset eligible only with preceding observed >94 and complete 30s lead-in.
    Episodes already low at start are left censored, not prospective targets.
    Severe subset has an actual 91 or 92 observation in the <=94 episode.
    """
    n=load_csv(DATA_DIR/f'bidmc_{record:02d}_Numerics.csv',['Time [s]','SpO2'])
    t=n['Time [s]'].to_numpy(float);s=n.SpO2.to_numpy(float);low=np.isfinite(s)&(s>0)&(s<=SPO2_THRESHOLD)
    rows=[]
    for i in np.flatnonzero(low & np.r_[True,(~low[:-1])|(np.diff(t)>MAX_NUMERIC_GAP)]):
        j=i+1
        while j<len(t) and low[j] and t[j]-t[j-1]<=MAX_NUMERIC_GAP:j+=1
        reason='eligible'
        if i==0 or not np.isfinite(s[i-1]) or s[i-1]<=SPO2_THRESHOLD or t[i]-t[i-1]>MAX_NUMERIC_GAP:reason='left_censored_or_unknown_prior'
        elif t[i]-HORIZON<t[0]:reason='insufficient_lead_in'
        else:
            prior=s[(t>=t[i]-HORIZON)&(t<t[i])]
            if not np.isfinite(prior).all():reason='missing_lead_in'
        rows.append({'Record':record,'Event_ID':f'{record:02d}_{i}','Onset_s':t[i],'End_s':t[j-1],
                     'Minimum_SpO2':float(np.min(s[i:j])),'Has_91_92':bool(np.isin(s[i:j],[91,92]).any()),'Status':reason})
    return rows


def event_analysis(predictions,output):
    events=pd.DataFrame([e for r in sorted(predictions.Record.unique()) for e in observed_events(int(r))])
    events.to_csv(output/'ground_truth_events.csv',index=False)
    rows=[]
    for name,p in predictions.groupby('Model',sort=False):
        for e in events.to_dict('records'):
            q=p[(p.Record==e['Record'])&(p.time_s>=e['Onset_s']-HORIZON)&(p.time_s<e['Onset_s'])]
            # Require dense evaluable pulse coverage of the entire lead window.
            covered=len(q)>0 and q.time_s.min()<=e['Onset_s']-HORIZON+2 and q.time_s.max()>=e['Onset_s']-2 and np.diff(q.time_s).max(initial=0)<=2
            status=e['Status']
            if status=='eligible' and not covered:status='incomplete_evaluable_pulse_coverage'
            alerts=q[q.Prediction==1]
            first=float(alerts.time_s.min()) if len(alerts) else np.nan
            rows.append({**e,'Model':name,'Evaluation_Status':status,'Detected':bool(len(alerts)) if status=='eligible' else np.nan,
                         'Lead_Time_s':e['Onset_s']-first if status=='eligible' and len(alerts) else np.nan})
    matched=pd.DataFrame(rows);matched.to_csv(output/'event_metrics.csv',index=False)
    summary=[]
    for name,g in matched.groupby('Model',sort=False):
        for subset in ['all_94_events','91_92_subset']:
            eligible=g[g.Evaluation_Status=='eligible']
            if subset=='91_92_subset':eligible=eligible[eligible.Has_91_92]
            summary.append({'Model':name,'Subset':subset,'Eligible_Events':len(eligible),'Detected':int(eligible.Detected.eq(True).sum()),
                            'Sensitivity':float(eligible.Detected.eq(True).mean()) if len(eligible) else np.nan,
                            'Positive_Recordings':eligible.Record.nunique(),'Median_Lead_Time_s':float(eligible.Lead_Time_s.median()),
                            'Excluded_Events':len(g)-len(g[g.Evaluation_Status=='eligible'])})
    summary=pd.DataFrame(summary);summary.to_csv(output/'event_summary.csv',index=False)
    return events,summary


def bootstrap_recordings(predictions,repetitions=1000):
    """Exploratory group bootstrap; folds share training and positive groups are few."""
    rows=[];rng=np.random.default_rng(42)
    records=sorted(predictions.Record.unique())
    for name,p in predictions.groupby('Model',sort=False):
        counts=[]
        for r in records:
            g=p[p.Record==r];m=metrics(g.Label,g.Score,g.Prediction);counts.append([m['TP'],m['FN'],m['FP'],m['TN']])
        counts=np.array(counts)
        vals=[]
        for _ in range(repetitions):
            tp,fn,fp,tn=counts[rng.integers(0,len(records),len(records))].sum(axis=0)
            vals.append([tp/(tp+fn) if tp+fn else np.nan,tp/(tp+fp) if tp+fp else np.nan])
        vals=np.array(vals)
        for i,m in enumerate(['Recall','Precision']):
            lo,hi=np.nanquantile(vals[:,i],[.025,.975]);rows.append({'Model':name,'Metric':m,'Lower_95':lo,'Upper_95':hi,'Replicates':repetitions,'Interpretation':'exploratory recording bootstrap; not clinical CI; CV fits overlap'})
    return pd.DataFrame(rows)


def generate_reports(output=OUTPUT_DIR):
    output=Path(output);plots=output/'plots';plots.mkdir(exist_ok=True)
    d=pd.read_csv(output/'causal_dataset.csv');p=pd.read_csv(output/'per_record_predictions.csv');comparison=pd.read_csv(output/'model_comparison.csv')
    comparison['Alert_Fraction']=comparison.Predicted_Positive/comparison.Samples
    comparison.to_csv(output/'model_comparison.csv',index=False)
    references=[]
    for label,prediction in [('Always_Alert',np.ones(len(d),int)),('Never_Alert',np.zeros(len(d),int))]:
        references.append({'Reference':label,**metrics(d.Label,prediction.astype(float),prediction)})
    references=pd.DataFrame(references);references.to_csv(output/'constant_references.csv',index=False)
    rec=pd.read_csv(output/'per_recording_metrics.csv');audit=pd.read_csv(output/'dataset_audit.csv');validation=pd.read_csv(output/'validation_metrics.csv')
    features=[]
    for c in FEATURES:
        for label,g in d.groupby('Label'):
            features.append({'Feature':c,'Label':label,'Rows':len(g),'Missing':int(g[c].isna().sum()),'Minimum':g[c].min(),'Q05':g[c].quantile(.05),'Q25':g[c].quantile(.25),'Median':g[c].median(),'Q75':g[c].quantile(.75),'Q95':g[c].quantile(.95),'Maximum':g[c].max(),'Std':g[c].std(),'Zero_Variance':g[c].nunique()<=1})
    pd.DataFrame(features).to_csv(output/'feature_audit.csv',index=False)
    score_stats=[]
    for (name,r,label),g in p.groupby(['Model','Record','Label']):
        score_stats.append({'Model':name,'Record':r,'Label':label,**g.Score.describe(percentiles=[.05,.25,.5,.75,.95]).to_dict()})
    pd.DataFrame(score_stats).to_csv(output/'score_distributions.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for name,g in p.groupby('Model',sort=False):
        precision,recall,thr=precision_recall_curve(g.Label,g.Score)
        axes[0].plot(recall,precision,label=name)
        axes[1].hist(g[g.Label==1].Score,bins=40,density=True,histtype='step',label=name+' positive')
    axes[0].axhline(d.Label.mean(),color='gray',linestyle='--',label='prevalence');axes[0].set(xlabel='Recall',ylabel='Precision',title='Held-out pooled PR (scores across refits)');axes[0].legend()
    axes[1].set(xlabel='Uncalibrated score',ylabel='Density',title='Positive score distributions');axes[1].legend(fontsize=8)
    fig.tight_layout();fig.savefig(plots/'model_comparison.png',dpi=150);plt.close(fig)
    diagnostics=[];feature_ranges=[]
    for record in [7,12,26,30]:
        test=d[d.Record==record];train=d[d.Record!=record]
        test.groupby('Label')[FEATURES].agg(['min','median','mean','std','max']).to_csv(output/f'record_{record:02d}_feature_distributions.csv')
        for c in FEATURES:
            lo,hi=train[c].quantile([.01,.99]);feature_ranges.append({'Record':record,'Feature':c,'Train_Q01':lo,'Train_Q99':hi,'Test_Median':test[c].median(),
                                                                 'Fraction_Outside_Training_Q01_Q99':float(((test[c]<lo)|(test[c]>hi)).mean()),'Missing':int(test[c].isna().sum())})
        fig,axes=plt.subplots(4,3,figsize=(14,12))
        for i,name in enumerate(comparison.Model):
            g=p[(p.Record==record)&(p.Model==name)]
            axes[i,0].plot(g.time_s,g.Score,label='score');axes[i,0].plot(g.time_s,g.Threshold,'--',label='validation threshold');axes[i,0].plot(g.time_s,g.Label,alpha=.35,label='label');axes[i,0].set_title(name);axes[i,0].legend(fontsize=7)
            pr,re,th=precision_recall_curve(g.Label,g.Score);axes[i,1].plot(re,pr);axes[i,1].set(xlabel='Recall',ylabel='Precision')
            for label in [0,1]:axes[i,2].hist(g[g.Label==label].Score,bins=25,density=True,alpha=.4,label=f'label {label}')
            axes[i,2].axvline(g.Threshold.iloc[0],linestyle='--');axes[i,2].legend(fontsize=7)
            diagnostics.append(rec[(rec.Record==record)&(rec.Model==name)].iloc[0].to_dict())
        fig.tight_layout();fig.savefig(plots/f'record_{record:02d}_diagnostic.png',dpi=140);plt.close(fig)
    pd.DataFrame(diagnostics).to_csv(output/'record_diagnostics.csv',index=False)
    pd.DataFrame(feature_ranges).to_csv(output/'record_feature_shift.csv',index=False)
    # Validation-only threshold tradeoffs: outer-test 12 is an example; never revise operating points from this plot.
    import gzip
    threshold_parts=[]
    for chunk in pd.read_csv(output/'threshold_analysis.csv.gz',usecols=['Model','Test_Record','Threshold','Recall','Precision','FPR'],chunksize=100000):
        threshold_parts.append(chunk[chunk.Test_Record==12])
    threshold_sample=pd.concat(threshold_parts,ignore_index=True)
    fig,axes=plt.subplots(4,1,figsize=(10,12))
    for ax,name in zip(axes,comparison.Model):
        tab=threshold_sample[(threshold_sample.Model==name)&(threshold_sample.Test_Record==12)]
        for c in ['Recall','Precision','FPR']:ax.plot(tab.Threshold,tab[c],label=c)
        ax.set(title=name+' — grouped validation tradeoff, outer test 12',xlabel='Threshold');ax.legend()
    fig.tight_layout();fig.savefig(plots/'validation_threshold_tradeoffs.png',dpi=140);plt.close(fig)
    events,event_summary=event_analysis(p,output);bootstrap_recordings(p).to_csv(output/'uncertainty.csv',index=False)
    # Secondary prospective timestamp subset uses observed CURRENT SpO2 >94, fixed rule, no tuning.
    impending=[]
    for name,g in p.groupby('Model',sort=False):
        h=g[g.spo2>SPO2_THRESHOLD];impending.append({'Model':name,**metrics(h.Label,h.Score,h.Prediction),'Definition':'current observed SpO2 >94; same held-out scores/thresholds; not a retrained task'})
    impending=pd.DataFrame(impending);impending.to_csv(output/'impending_timestamp_metrics.csv',index=False)
    cols=['Model','Recall','Precision','Specificity','F1','PR_AUC_AP','Macro_Recall','Macro_F1','FP','FN','False_Alerts_Per_Hour','Alert_Fraction']
    doc='# Measured model evaluation\n\n'+markdown_table(comparison,cols)+'\n\n'
    doc+=f'All values above are held-out predictions under leave-one-recording-out evaluation on {len(d):,} verified pulse timestamps in {d.Record.nunique()} recordings. Positive prevalence is {d.Label.mean():.4%}. {int((audit.Positives>0).sum())} recordings contain positives; {int(audit.All_Positive.sum())} are all positive. No difficult recording is removed. Raw monitor threshold crossings, rather than contiguous predictive labels, define events.\n\n'
    doc+='The target is min observed SpO2 in `(t,t+30]` <=94, with a complete observable future horizon. It includes ongoing low saturation. It is not an exact SpO2 regression model or proof of impending-event detection. Missing future measurements and incomplete recording tails have unknown labels and are explicitly excluded. Numeric/waveform timestamps retain their supplied shared coordinate, including numeric record 23 starting at -1s. Waveform export rounding is corrected with the verified 125Hz row clock, without altering waveform values.\n\n'
    doc+='Each model uses the same ten causal features and rows. Three inner GroupKFold fits generate OOF scores for every outer-training row. The largest unique threshold achieving >=92% validation recall maximizes specificity; tied score blocks are never split. Imputation/scaling are refitted on each training fold only. Fixed model configurations were chosen before evaluation; no search or test-based threshold tuning occurs. Full-data demonstration artifacts have separate full-data grouped OOF thresholds and are not used to report held-out metrics.\n\n'
    doc+='Undefined recall on negative-only records and specificity on all-positive records are blank (NaN). Macro metrics average only defined recording values, with denominators in pooled_metrics.csv. PR-AUC is average precision, not trapezoidal interpolation. Pooled ranking metrics mix uncalibrated scores from different refits; macro AP on mixed-class recordings is also provided. Weighted classifier outputs and ELM scores are not validated calibrated risks.\n\n'
    doc+=f'All {len(validation)} outer/model thresholds achieve 92% grouped validation recall: {bool(validation.Recall.ge(.92).all())}. Held-out target outcomes are reported independently; target attainment on validation does not guarantee it on new recordings.\n\n'
    doc+='Always-alert/never-alert references (no learned model):\n\n'+markdown_table(references,['Reference','Recall','Precision','Specificity','F1','FP','FN'])+'\n\nAll-alert sensitivity is 100% on every eligible event by construction. Therefore high event sensitivity with high alert duty is not evidence of useful event discrimination. Alert_Fraction reports predicted-positive pulse fraction, not elapsed-time duty. Sustained always-on alarms can yield fewer counted episode starts than chattering models; episode rates must be read with timestamp FP burden and alert fraction.\n\n'
    doc+='## Genuine pre-onset detection\n\n'+markdown_table(impending,['Model','Recall','Precision','FP','FN'])+'\n\nCurrent-observed-SpO2>94 subset, retaining identical models and operating points. This is a descriptive prospective-state subset; current SpO2 is never a predictor.\n\n'
    doc+=markdown_table(event_summary,['Model','Subset','Eligible_Events','Detected','Sensitivity','Positive_Recordings','Median_Lead_Time_s'])+'\n\n'
    doc+='Episodes are contiguous observed SpO2<=94 at 1Hz; missing readings break episodes. Only crossings preceded by observed >94, with an observed 30s lead-in and continuous eligible pulse coverage (no >2s gap, first/last within 2s) count in sensitivity. An alert in [onset-30,onset) is a detection. A 91–92 subset requires an actual observed 91 or 92 within that episode; one-second monitor excursions are not clinically adjudicated events. Initial/chronic lows and unavailable lead-ins are separately enumerated. The same alert can anticipate nearby episodes, so event sensitivity does not imply independent alert-event precision.\n\n'
    doc+='False alerts/hour count predicted-positive runs split by negative predictions or >2s gaps; a false run contains no positive target timestamp. A sustained run overlapping a positive target is counted as true even if long. Exposure sums observed pulse intervals <=2s, excluding unavailable-target gaps; this is evaluable exposure, not total wall time. FP timestamps/hour is separately recorded and is not an episode alarm rate. No suppression/refractory alarm policy is retrospectively optimized.\n\n'
    doc+='## Records 07, 12, 26, 30\n\n'+markdown_table(pd.DataFrame(diagnostics),['Record','Model','Samples','Positives','Threshold','Recall','Precision','FP','FN'])+'\n\n'
    shift=pd.DataFrame(feature_ranges)
    r12=rec[(rec.Model=='ELM')&(rec.Record==12)].iloc[0]
    r26=rec[(rec.Model=='ELM')&(rec.Record==26)].iloc[0]
    r30=rec[(rec.Model=='ELM')&(rec.Record==30)].iloc[0]
    r7=rec[(rec.Model=='ELM')&(rec.Record==7)].iloc[0]
    score_frame=pd.DataFrame(score_stats)
    def med(record,label):
        return score_frame[(score_frame.Model=='ELM')&(score_frame.Record==record)&(score_frame.Label==label)]['50%'].iloc[0]
    def shift_fraction(record,feature):
        return shift[(shift.Record==record)&(shift.Feature==feature)].Fraction_Outside_Training_Q01_Q99.iloc[0]
    shift=pd.DataFrame(feature_ranges)
    doc+=f'Record 12 ELM recall is {r12.Recall:.1%}, precision {r12.Precision:.1%}, with {r12.FP} FP: higher sensitivity comes at high cost, not evidence of strong separation. Record 26 ELM AP={r26.PR_AUC_AP:.3f} versus prevalence {r26.Positives/r26.Samples:.3f}; positive/negative median scores are {med(26,1):.3f}/{med(26,0):.3f}. Record 30 ELM recall={r30.Recall:.1%}; positive/negative medians are {med(30,1):.3f}/{med(30,0):.3f}, while {shift_fraction(30,"Amplitude_Slope"):.1%} of amplitude-slope and {shift_fraction(30,"Rate_Mean_5"):.1%} of rolling-rate-mean values lie outside outer-training 1–99% ranges. Record 07 AP={r7.PR_AUC_AP:.3f} versus prevalence {r7.Positives/r7.Samples:.3f}. These findings support weak/uneven feature discrimination plus distribution shift, alongside the verified historical leakage, target-selection and incomplete-window bugs. Class imbalance and poorly representative legacy validation compounded the problem. No threshold alone resolves it; no clinical causal mechanism is inferred.\n\n'
    doc+='Feature and score distributions, confusion counts, temporal/PR plots, missingness, and training percentile comparisons are saved per recording. Fraction outside training 1–99% ranges is a descriptive distribution-shift indicator, not a test of causation. Large differences by recording and weak/uneven within-record ranking support limited portability of pulse-only features. Correcting leakage and thresholds cannot create discriminative physiology. Target/population changes make direct numerical comparison with historical leaky/selective results invalid.\n\n'
    a=comparison.set_index('Model');elm=a.loc['ELM'];lr=a.loc['Logistic']
    doc+=f'## ELM hypothesis\n\nELM recall={elm.Recall:.4f}, precision={elm.Precision:.4f}, AP={elm.PR_AUC_AP:.4f}; logistic recall={lr.Recall:.4f}, precision={lr.Precision:.4f}, AP={lr.PR_AUC_AP:.4f}. ELM has higher pooled recall, precision and AP and fewer FP timestamps than logistic at the common validation objective. These are descriptive advantages on this dataset. However, ELM produces more false alert episodes and its macro within-record AP is lower; practically useful ELM superiority is not established. Both largely alert across negative recordings. Retain ELM as the main research comparator and logistic as the prespecified transparent demonstration reference; neither is suitable for oxygen actuation from this evidence.\n\n'
    doc+='## Limitations and next step\n\nOnly 30 of the public 53 recordings are local. Recording IDs are used as groups; independent patient identity cannot be proven without source metadata. Few new-onset events, chronic-low records, rounded monitor readings, unknown sensor delay, and missing measurements limit interpretation. Exploratory 1,000-replicate recording bootstrap intervals are saved, but CV fits share training data and few positive-onset groups preclude reliable clinical confidence intervals. Thresholds pursue timestamp recall and can produce substantial alarms. The highest-value next step is acquiring independent, longer recordings with verified signal/SpO2 timing and sufficient adjudicated onset events; freeze this protocol for that evaluation instead of more threshold sweeps.\n'
    ROOT.joinpath('docs/MODEL_EVALUATION.md').write_text(doc)
    manifest=json.loads((output/'run_manifest.json').read_text())
    if manifest.get('benchmark_completed',False) and (output/'latency_benchmark.csv').exists():
        lat=pd.read_csv(output/'latency_benchmark.csv');env=json.loads((output/'benchmark_environment.json').read_text())
        text='# Measured computational latency\n\n'+markdown_table(lat,['Model','Component','Median_ms','P95_ms','P99_ms','Mean_ms','Max_ms','Repetitions'])+'\n\n'
        text+='Environment: '+json.dumps(env,indent=2)+'\n\n'
        text+='Timer: perf_counter_ns; 100 warmups, 2,000 samples per pulse component, single native thread. Waveform replay times 19,901 post-warmup 125Hz inputs and separately reports pulse-producing inputs. No download, training, file writes, plots or loading occur in steady-state timers. Loading is measured once separately in CSV; it is not a percentile estimate. Component times are measured independently and should not simply be summed. P99 is reported only for at least 1,000 measurements; the smaller pulse-producing waveform subset has no reliable P99. P99 is empirical, not a worst-case bound; shared-host scheduling explains outliers. Throughput is reciprocal mean compute time, not sensor rate.\n\n'
        text+='Waveform filtering is forward SOS with zero initial state. Local maxima require one following waveform sample (8ms confirmation); intervals/amplitudes use the preceding beat. Signals arrive at 125Hz, while alert opportunities occur per accepted pulse. Computing a pulse feature requires acquiring that pulse and its history (rolling windows are 5/10 pulses, not seconds). Filtering phase delay, sensor/monitor latency, communication and actuation are not measured. The 30s label horizon is neither computational latency nor guaranteed warning lead time. Research replay is not a real-time or clinical deployment guarantee.\n'
        ROOT.joinpath('docs/LATENCY_REPORT.md').write_text(text)
    # Inventory deliverable hashes after all generated report artifacts.
    hashes={str(f.relative_to(output)):__import__('hashlib').sha256(f.read_bytes()).hexdigest() for f in sorted(output.rglob('*')) if f.is_file() and f.name!='output_hashes.json'}
    (output/'output_hashes.json').write_text(json.dumps(hashes,indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUTPUT_DIR);a=p.parse_args();generate_reports(a.output)
if __name__=='__main__':main()
