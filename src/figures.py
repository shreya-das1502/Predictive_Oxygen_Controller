"""Reproducible scientific figures from one evaluated run; no tuning or fabricated data."""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', '/tmp/oxygen-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
from scipy.signal import sosfilt
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve, roc_auc_score
from threadpoolctl import threadpool_limits
from . import config
from .dataset import load_csv
from .model import FittedModel
from .preprocessing import PulseDetector, WaveformState, canonical_waveform_times
from .provenance import fingerprint

COLORS = {'ELM':'#0072B2','Logistic':'#D55E00','GaussianNB':'#009E73','Tree':'#CC79A7'}
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                     'figure.dpi':120,'savefig.dpi':220,'pdf.fonttype':42,'font.family':'DejaVu Sans'})


def plotted_metrics(comparison):
    rows=[]
    for _,r in comparison.iterrows():
        for metric in ['Recall','Precision','F1','Specificity','PR_AUC_AP','Macro_Recall','Macro_Precision','Macro_F1']:
            rows.append({'Model':r.Model,'Metric':metric,'Value':r[metric],
                         'Scope':'macro of defined per-recording values' if metric.startswith('Macro') else 'pooled held-out timestamps'})
    return pd.DataFrame(rows)


class FigureRun:
    def __init__(self,output):
        self.output=Path(output);self.directory=self.output/'figures';self.directory.mkdir(exist_ok=True)
        self.manifest=json.loads((self.output/'run_manifest.json').read_text())
        self.target=self.manifest['target_definition']['name']
        self.d=pd.read_csv(self.output/'causal_dataset.csv')
        self.p=pd.read_csv(self.output/'per_record_predictions.csv')
        self.c=pd.read_csv(self.output/'model_comparison.csv')
        self.r=pd.read_csv(self.output/'per_recording_metrics.csv')
        self.index=[];self.raw_cache={}
        index_path=self.directory/'INDEX.md'
        if index_path.exists():
            for name,caption in re.findall(r'- \[([^\]]+)\.png\]\([^)]+\) / \[PDF\]\([^)]+\): (.*)',index_path.read_text()):
                self.index.append({'filename':name,'interpretation':caption})
        if fingerprint(self.output/'causal_dataset.csv')!=self.manifest['dataset_sha256']:
            raise ValueError('Dataset does not match the evaluated run manifest')
        for model,g in self.p.groupby('Model'):
            if set(g.sample_id)!=set(self.d.sample_id) or g.sample_id.duplicated().any():raise ValueError('Unmatched plotted evaluation rows')
            tp=((g.Label==1)&(g.Prediction==1)).sum();fp=((g.Label==0)&(g.Prediction==1)).sum()
            row=self.c.set_index('Model').loc[model]
            if tp!=row.TP or fp!=row.FP:raise ValueError('Plotted metrics disagree with saved predictions')

    def save(self,fig,name,caption,data=None):
        if not fig.get_constrained_layout():fig.tight_layout()
        for ext in ['png','pdf']:fig.savefig(self.directory/f'{name}.{ext}',bbox_inches='tight')
        plt.close(fig)
        if data is not None:data.to_csv(self.directory/f'{name}.csv',index=False)
        self.index=[entry for entry in self.index if entry['filename']!=name]
        self.index.append({'filename':name,'interpretation':caption})

    def raw(self,record):
        record=int(record)
        if record not in self.raw_cache:
            for suffix in ['Signals','Numerics']:
                path=config.DATA_DIR/f'bidmc_{record:02d}_{suffix}.csv'
                if self.manifest['source_hashes'].get(str(path))!=fingerprint(path):raise ValueError('Raw figure input differs from evaluated source')
            s=load_csv(config.DATA_DIR/f'bidmc_{record:02d}_Signals.csv',['Time [s]','PLETH','RESP'])
            n=load_csv(config.DATA_DIR/f'bidmc_{record:02d}_Numerics.csv',['Time [s]','SpO2'])
            t=canonical_waveform_times(s['Time [s]'])
            self.raw_cache[record]=(s,n,t)
        return self.raw_cache[record]

    def model_plots(self):
        values=plotted_metrics(self.c)
        fig,axes=plt.subplots(2,4,figsize=(15,7))
        for ax,metric in zip(axes.ravel(),values.Metric.unique()):
            q=values[values.Metric==metric]
            ax.bar(q.Model,q.Value,color=[COLORS[m] for m in q.Model]);ax.set_ylim(0,1)
            ax.set(title=metric.replace('_',' '),ylabel='Fraction [0–1]');ax.tick_params(axis='x',rotation=25)
        fig.suptitle(f'{self.target}: pooled timestamp metrics and separately labelled macro metrics',y=1.02)
        self.save(fig,'model_metrics','Pooled and macro metrics have different denominators; undefined recording metrics are excluded from macro means. High recall with low specificity is not a useful detector.',values)
        fig,axes=plt.subplots(1,2,figsize=(12,5));curves=[]
        for model,g in self.p.groupby('Model',sort=False):
            pr,re,th=precision_recall_curve(g.Label,g.Score);fpr,tpr,roc_th=roc_curve(g.Label,g.Score)
            axes[0].plot(re,pr,label=f'{model} AP={average_precision_score(g.Label,g.Score):.3f}',color=COLORS[model])
            axes[1].plot(fpr,tpr,label=f'{model} AUC={roc_auc_score(g.Label,g.Score):.3f}',color=COLORS[model])
            curves.extend({'Model':model,'Curve':'PR','X':x,'Y':y} for x,y in zip(re,pr))
            curves.extend({'Model':model,'Curve':'ROC','X':x,'Y':y} for x,y in zip(fpr,tpr))
        axes[0].axhline(self.d.Label.mean(),color='gray',ls='--',label='Positive prevalence')
        axes[1].plot([0,1],[0,1],color='gray',ls='--',label='Chance reference')
        for ax in axes:ax.set(xlim=(0,1),ylim=(0,1));ax.legend(fontsize=8)
        axes[0].set(xlabel='Recall',ylabel='Precision',title='Held-out PR curves')
        axes[1].set(xlabel='False-positive rate',ylabel='Sensitivity',title='Held-out ROC curves')
        self.save(fig,'ranking_curves','Pooled curves combine uncalibrated scores from different refits. AP means average precision; ROC can be misleading under class imbalance.',pd.DataFrame(curves))
        fig,axes=plt.subplots(1,4,figsize=(14,4))
        for ax,(_,r) in zip(axes,self.c.iterrows()):
            matrix=np.array([[r.TN,r.FP],[r.FN,r.TP]],int)
            ax.imshow(matrix,cmap='Blues',vmin=0,vmax=self.c[['TP','TN','FP','FN']].to_numpy().max())
            for (i,j),v in np.ndenumerate(matrix):ax.text(j,i,str(v),ha='center',va='center')
            ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['No alert','Alert'],yticklabels=['Negative','Positive'],xlabel='Predicted decision',ylabel='Observed future target',title=r.Model)
        self.save(fig,'confusion_matrices','Counts use the same held-out rows per model, with observed target on rows and predicted decision on columns.',self.c[['Model','TN','FP','FN','TP']])
        fig,axes=plt.subplots(1,2,figsize=(11,8),layout='constrained')
        cmap=plt.get_cmap('viridis').copy();cmap.set_bad('#d5d5d5')
        for ax,metric in zip(axes,['Recall','Precision']):
            pivot=self.r.pivot(index='Record',columns='Model',values=metric).reindex(columns=list(COLORS))
            im=ax.imshow(pivot.to_numpy(),aspect='auto',vmin=0,vmax=1,cmap=cmap)
            ax.set(xticks=np.arange(4),xticklabels=list(COLORS),yticks=np.arange(len(pivot)),yticklabels=pivot.index,xlabel='Model',ylabel='Held-out recording',title=metric+' (gray = undefined)');ax.tick_params(axis='x',rotation=25)
        fig.colorbar(im,ax=list(axes),label='Fraction [0–1]',shrink=.8)
        self.save(fig,'recording_performance','Negative-only recordings have undefined recall, not zero. All-positive eligible snippets can inflate simple macro summaries.',self.r)
        fig,axes=plt.subplots(1,3,figsize=(14,4))
        for ax,metric,title in zip(axes,['FP','False_Alerts_Per_Hour','Alert_Fraction'],['FP timestamp count','False alarm episodes / evaluable hour','Predicted-positive pulse fraction']):
            ax.bar(self.c.Model,self.c[metric],color=[COLORS[m] for m in self.c.Model]);ax.set(title=title,ylabel=title);ax.set_ylim(bottom=0);ax.tick_params(axis='x',rotation=25)
        axes[2].set_ylim(0,1)
        self.save(fig,'false_alarm_burden','Episode starts and FP timestamps measure different burdens. Sustained alerts can have few starts but very poor specificity. Exposure excludes unevaluable periods.',self.c[['Model','FP','False_Alerts_Per_Hour','Alert_Fraction','Exposure_Hours']])

    def dataset_plots(self):
        audit=pd.read_csv(self.output/'dataset_audit.csv');stages=pd.read_csv(self.output/'preprocessing_stages.csv')
        fig,axes=plt.subplots(1,3,figsize=(15,4))
        x=np.arange(len(audit));axes[0].bar(x,audit.Negatives,label='Negative',color='#999999');axes[0].bar(x,audit.Positives,bottom=audit.Negatives,label='Positive',color='#D55E00')
        axes[0].set(xticks=x,xticklabels=audit.Record,xlabel='Recording',ylabel='Eligible pulse timestamps',title=f'{self.target}: class counts');axes[0].tick_params(axis='x',labelsize=7);axes[0].legend()
        for _,r in audit.iterrows():
            if r.Rows==0:axes[0].text(list(audit.Record).index(r.Record),0,'empty',rotation=90,fontsize=7)
        pulse=int(audit.Pulse_Rows.sum());valid=len(self.d);positive=int(self.d.Label.sum())
        axes[1].bar(['Extracted pulses','After rolling','Eligible targets'],[pulse,pulse,valid],color=['#999999','#999999','#0072B2'])
        axes[1].set(ylabel='Pulse timestamps',title='Retained rows by stage');axes[1].tick_params(axis='x',rotation=20)
        reason={}
        for path in sorted(self.output.glob('raw_pulses_*.csv')):
            for key,count in pd.read_csv(path).Target_Status.value_counts().items():reason[key]=reason.get(key,0)+int(count)
        axes[2].barh(list(reason),list(reason.values()),color='#56B4E9');axes[2].set(xlabel='Pulse timestamps',title='Target validity / eligibility')
        self.save(fig,'dataset_and_horizon','Rolling features retain every pulse. Unknown future labels and current-ineligible states are excluded, not labelled negative; positive counts among unknown targets cannot be inferred.',audit)
        pd.DataFrame([{'Stage':'extracted_pulses','Rows':pulse,'Known_Positive_Targets':positive,'Unknown_or_ineligible':pulse-valid},
                      {'Stage':'rolling_features','Rows':pulse,'Known_Positive_Targets':positive,'Unknown_or_ineligible':pulse-valid},
                      {'Stage':'eligible_targets','Rows':valid,'Known_Positive_Targets':positive,'Unknown_or_ineligible':0}]).to_csv(self.directory/'stage_counts.csv',index=False)
        quality=pd.read_csv(self.output/'raw_data_quality.csv');records=sorted(quality.Record.unique());rows=[]
        for r in records:
            qs=quality[quality.Record==r];n=qs[qs.Source=='numerics'].iloc[0];s=qs[qs.Source=='waveform'].iloc[0]
            mn=json.loads(n.Missing_JSON);ms=json.loads(s.Missing_JSON)
            rows.append({'Record':r,'SpO2 missing fraction':mn.get('SpO2',0)/n.Rows,'PPG missing fraction':ms.get('PLETH',0)/s.Rows,'Respiration missing fraction':ms.get('RESP',0)/s.Rows,'Rounded waveform duplicate times':s.Duplicate_Timestamps})
        q=pd.DataFrame(rows);fig,axes=plt.subplots(1,2,figsize=(10,7))
        im=axes[0].imshow(q.iloc[:,1:4].to_numpy(),aspect='auto',vmin=0,vmax=max(.01,q.iloc[:,1:4].to_numpy().max()),cmap='magma')
        axes[0].set(xticks=[0,1,2],xticklabels=['SpO₂','PPG','Respiration'],yticks=np.arange(len(q)),yticklabels=q.Record,title='Raw missingness fractions',xlabel='Observed channel',ylabel='Recording');fig.colorbar(im,ax=axes[0],label='Missing fraction')
        axes[1].barh(q.Record.astype(str),q.iloc[:,-1],color='#999999');axes[1].set(xlabel='Duplicate displayed times (count)',ylabel='Recording',title='Rounded CSV times: samples retained')
        self.save(fig,'missingness_and_clock','Missing SpO₂ invalidates future windows. Rounded waveform times are reconstructed only when consistent with the 125Hz row clock; duplicate displayed times are not deleted.',q)
        pulses=pd.concat([pd.read_csv(p) for p in sorted(self.output.glob('raw_pulses_*.csv'))],ignore_index=True)
        fig,axes=plt.subplots(1,3,figsize=(12,4))
        for ax,col,unit in zip(axes,['Pulse_Interval_s','Pulse_Rate_BPM','PPG_Amplitude'],['s','beats/min','a.u.']):
            ax.hist(pulses[col],bins=50,color='#0072B2');ax.set(xlabel=f'{col} ({unit})',ylabel='Accepted pulse count',title='Signal-quality proxy: '+col)
        self.save(fig,'pulse_quality_proxies','Intervals are constrained to 0.30–2s. Pulse rate is derived, not independent ground truth. Morphology/dicrotic peaks and clinical signal quality are not validated by these ranges.',pulses[['Record','time_s','Pulse_Interval_s','Pulse_Rate_BPM','PPG_Amplitude']])

    def features(self):
        fig,axes=plt.subplots(2,5,figsize=(16,7))
        for ax,c in zip(axes.ravel(),config.FEATURES):
            groups=[self.d[self.d.Label==label][c].dropna().to_numpy() for label in [0,1]]
            ax.boxplot(groups,showfliers=False,tick_labels=['Negative','Positive']);ax.set(title=c,ylabel='s' if 'Interval' in c else ('beats/min' if 'Rate' in c else ('a.u./s' if c=='Amplitude_Slope' else 'a.u.')))
        self.save(fig,'feature_distributions','Boxes show median and quartiles; whiskers use the standard 1.5 IQR rule, with outlier markers omitted for readability. Full distributions remain in the dataset. These are exploratory held-out descriptions.')
        corr=self.d[config.FEATURES].corr(method='spearman');fig,ax=plt.subplots(figsize=(10,8))
        im=ax.imshow(corr,vmin=-1,vmax=1,cmap='coolwarm');ax.set(xticks=np.arange(len(corr)),yticks=np.arange(len(corr)),xticklabels=corr.columns,yticklabels=corr.columns,title='Descriptive Spearman feature correlation');ax.tick_params(axis='x',rotation=90);fig.colorbar(im,ax=ax,label='Spearman correlation [−1,1]')
        self.save(fig,'feature_correlation','Pulse rate and interval are mathematically related. Correlation is descriptive, not causation or a basis for post-hoc feature selection.',corr.reset_index())
        fig,axes=plt.subplots(1,3,figsize=(13,4));rows=[]
        for ax,record in zip(axes,[12,26,30]):
            train=self.d[self.d.Record!=record];test=self.d[self.d.Record==record]
            rates=[]
            for c in config.FEATURES:
                lo,hi=train[c].quantile([.01,.99]);fraction=((test[c]<lo)|(test[c]>hi)).mean();rates.append(fraction)
                rows.append({'Record':record,'Feature':c,'Training_Q01':lo,'Training_Q99':hi,'Test_Outside_Fraction':fraction})
            ax.barh(config.FEATURES,rates,color='#E69F00');ax.set(xlim=(0,1),xlabel='Fraction outside training 1–99% range',title=f'Held-out record {record:02d}')
        self.save(fig,'recording_feature_shift','Training ranges exclude the plotted recording. This is exploratory shift evidence; it has not been used to change the evaluated models.',pd.DataFrame(rows))

    def threshold_plots(self):
        pieces=[]
        for chunk in pd.read_csv(self.output/'threshold_analysis.csv.gz',chunksize=100000):
            pieces.append(chunk[chunk.Test_Record==12])
        data=pd.concat(pieces,ignore_index=True)
        fig,axes=plt.subplots(4,2,figsize=(12,12))
        for i,model in enumerate(COLORS):
            q=data[data.Model==model].sort_values('Threshold')
            for col in ['Recall','Precision','FPR']:axes[i,0].plot(q.Threshold,q[col],label=col)
            axes[i,0].set(xlabel='Validation score threshold',ylabel='Fraction [0–1]',ylim=(0,1),title=model+' — OOF training records; test 12 excluded');axes[i,0].legend()
            for label,color in [(0,'#777777'),(1,'#D55E00')]:
                vals=self.p[(self.p.Model==model)&(self.p.Label==label)].Score
                axes[i,1].hist(vals,bins=50,density=True,histtype='step',color=color,label=f'Observed label {label}')
            axes[i,1].set(xlabel='Uncalibrated held-out score',ylabel='Density',title=model+' — positive/negative overlap');axes[i,1].legend()
        self.save(fig,'threshold_and_scores','Threshold curves use only grouped OOF training predictions for the prespecified outer record 12. Held-out score distributions are diagnostic; no operating point is chosen from them.',data)

    def temporal(self,record):
        s,n,t=self.raw(record);fig,axes=plt.subplots(5,1,figsize=(13,11),sharex=True)
        for ax,model in zip(axes[:4],COLORS):
            g=self.p[(self.p.Record==record)&(self.p.Model==model)]
            ax.plot(g.time_s,g.Score,color=COLORS[model],label='Held-out score');ax.plot(g.time_s,g.Threshold,'k--',label='Training-OOF threshold')
            bad=g[(g.Label==0)&(g.Prediction==1)];miss=g[(g.Label==1)&(g.Prediction==0)]
            ax.scatter(bad.time_s,bad.Score,color='#CC79A7',s=7,label='FP timestamp');ax.scatter(miss.time_s,miss.Score,color='#D55E00',s=12,marker='x',label='FN timestamp')
            ax.set(ylabel='Uncalibrated score',title=f'{model}: {len(bad)} FP / {len(miss)} FN');ax.legend(fontsize=8,ncol=4)
            twin=ax.twinx();twin.plot(g.time_s,g.Label,color='gray',alpha=.22);twin.set(ylim=(-.05,1.05),yticks=[0,1],ylabel='Observed label')
        axes[4].step(n['Time [s]'],n.SpO2,where='post',color='#009E73',label='Observed SpO₂');axes[4].axhline(config.SPO2_THRESHOLD,ls='--',color='black',label='Target threshold');axes[4].set(xlabel='Source recording time (s)',ylabel='SpO₂ (%)');axes[4].legend()
        fig.suptitle(f'{self.target}: held-out record {record:02d}',y=1.01)
        self.save(fig,f'record_{record:02d}_temporal','Scores and thresholds are held-out/validation quantities; gray labels and SpO₂ are ground truth. A missed positive timestamp is not necessarily a missed event.',self.p[self.p.Record==record])

    def examples(self):
        p=self.p[self.p.Model=='ELM'].sort_values(['Record','time_s'])
        selections=[]
        candidates={'negative':p[(p.Label==0)&(p.spo2>config.SPO2_THRESHOLD)&(p.time_s>=60)],
                    'positive':p[(p.Label==1)&(p.time_s>=60)],
                    'true_positive':p[(p.Label==1)&(p.Prediction==1)&(p.time_s>=60)],
                    'false_positive':p[(p.Label==0)&(p.Prediction==1)&(p.time_s>=60)],
                    'false_negative':p[(p.Label==1)&(p.Prediction==0)&(p.time_s>=60)]}
        for name,candidate in candidates.items():
            if len(candidate):selections.append((name,candidate.iloc[0]))
        events=pd.read_csv(self.output/'ground_truth_events.csv')
        severe=events[events.Qualifying_Onset&events.Has_91_92&(events.Onset_s>=config.HORIZON)].sort_values(['Record','Onset_s'])
        if len(severe):
            e=severe.iloc[0];c=p[(p.Record==e.Record)&(p.time_s<e.Onset_s)&(p.time_s>=e.Onset_s-config.HORIZON)]
            if len(c):selections.append(('observed_91_92_episode',c.iloc[len(c)//2]))
        grouped_selections={}
        for name,row in selections:
            key=(int(row.Record),float(row.time_s))
            if key not in grouped_selections:grouped_selections[key]=([name],row)
            else:grouped_selections[key][0].append(name)
        selections=[('_and_'.join(names),row) for names,row in grouped_selections.values()]
        choices=[]
        for name,row in selections:
            record=int(row.Record);anchor=float(row.time_s);s,n,t=self.raw(record);lo=max(0,anchor-10);hi=anchor+config.HORIZON+5
            mask=(t>=lo)&(t<=hi);ng=n[n['Time [s]'].between(lo,hi)]
            fig,axes=plt.subplots(4,1,figsize=(12,9),sharex=True)
            axes[0].plot(t[mask],s.PLETH.to_numpy()[mask],color='#0072B2');axes[0].set(ylabel='Raw PPG (a.u.)')
            axes[1].plot(t[mask],s.RESP.to_numpy()[mask],color='#009E73');axes[1].set(ylabel='Raw respiration (a.u.)')
            axes[2].step(ng['Time [s]'],ng.SpO2,where='post',label='Observed monitor SpO₂',color='#009E73');axes[2].scatter(ng['Time [s]'],ng.SpO2,s=9,color='#009E73');axes[2].axhline(config.SPO2_THRESHOLD,color='black',ls='--',label=f'Target ≤{config.SPO2_THRESHOLD:g}%');axes[2].set(ylabel='SpO₂ (%)');axes[2].legend(fontsize=8)
            q=p[(p.Record==record)&p.time_s.between(lo,hi)];axes[3].plot(q.time_s,q.Score,label='Held-out ELM score',color=COLORS['ELM']);axes[3].plot(q.time_s,q.Threshold,'k--',label='OOF threshold');axes[3].scatter([anchor],[row.Score],marker='x',s=55,color='#D55E00' if row.Label!=row.Prediction else '#009E73',label='Decision at prediction time',zorder=5);axes[3].set(xlabel='Source recording time (s)',ylabel='Uncalibrated score');axes[3].legend(fontsize=8)
            for ax in axes:ax.axvline(anchor,color='black',lw=1);ax.axvspan(anchor,anchor+config.HORIZON,color='#56B4E9',alpha=.12);ax.set_xlim(lo,hi)
            fig.suptitle(f'{self.target} / {name}: record {record:02d}, t={anchor:.3f}s; observed label={int(row.Label)}, alert={int(row.Prediction)}\nBlack line = prediction time; blue shading = strictly future horizon',y=1.02)
            self.save(fig,'example_'+name,'Selection: earliest matching ELM timestamp at ≥60s ordered by record/time; severe episode uses earliest qualifying 91/92 episode and its middle pre-onset opportunity. Negative does not imply clinical normality. Signals/SpO₂ are measured, score is predicted; synchronization and sensor delay remain unverified.')
            choices.append({'Example':name,'Record':record,'Prediction_Time_s':anchor,'Label':int(row.Label),'Prediction':int(row.Prediction),'Selection':'earliest eligible match, or earliest qualifying severe episode'})
        pd.DataFrame(choices).to_csv(self.directory/'example_selection.csv',index=False)
        self.peak_plot(12)

    def peak_plot(self,record):
        s,n,t=self.raw(record);filtered=sosfilt(WaveformState().sos,s.PLETH.to_numpy());detector=PulseDetector();peaks=[];accepted=[]
        for time,value in zip(t,filtered):
            before=detector.candidates;rejected=detector.rejected_intervals;row=detector.update(float(time),float(value))
            if detector.candidates>before:peaks.append((detector.last_peak,detector.rejected_intervals>rejected))
            if row is not None:accepted.append(row)
        peaks=pd.DataFrame(peaks,columns=['peak_time_s','Rejected_Interval']);a=pd.DataFrame(accepted)
        peaks.to_csv(self.directory/'ppg_candidates.csv',index=False)
        lo,hi=60,72;mask=(t>=lo)&(t<=hi);q=peaks[peaks.peak_time_s.between(lo,hi)];r=a[a.time_s.between(lo,hi)]
        fig,axes=plt.subplots(4,1,figsize=(12,9),sharex=True)
        axes[0].plot(t[mask],s.PLETH.to_numpy()[mask],label='Raw PPG',color='#777777');axes[0].set(ylabel='PPG (a.u.)');axes[0].legend()
        axes[1].plot(t[mask],filtered[mask],label='Causal SOS-filtered PPG',color='#0072B2')
        for rejected,color,marker,label in [(False,'#009E73','o','Accepted pulse candidate'),(True,'#D55E00','x','Rejected interval >2s')]:
            part=q[q.Rejected_Interval==rejected];idx=np.round(part.peak_time_s*125).astype(int)
            axes[1].scatter(part.peak_time_s,filtered[idx],color=color,marker=marker,label=label)
        axes[1].set(ylabel='Filtered PPG (a.u.)');axes[1].legend(fontsize=8)
        axes[2].plot(r.time_s,r.Pulse_Interval_s,'o-',color='#0072B2');axes[2].set(ylabel='Pulse interval (s)')
        axes[3].plot(r.time_s,r.Pulse_Rate_BPM,'o-',color='#D55E00');axes[3].set(ylabel='Pulse rate (beats/min)',xlabel='Source recording time (s)')
        fig.suptitle(f'Record {record:02d}: existing PPG pulse detector at 125Hz; one-sample confirmation',y=1.01)
        self.save(fig,'ppg_pulse_detection','These are PPG pulse maxima, not ECG R peaks. ECG channels exist but are not used by this pipeline. No new detector was introduced. Rejected >2s intervals are marked if present; no rejected points are invented. Physiological plausibility ranges are not peak annotation validation.',a)

    def events_latency(self):
        e=pd.read_csv(self.output/'event_summary.csv');m=pd.read_csv(self.output/'event_metrics.csv')
        fig,axes=plt.subplots(1,2,figsize=(12,5));subset=e[e.Subset=='all_qualifying_onsets']
        axes[0].bar(subset.Model,subset.Sensitivity,color=[COLORS[n] for n in subset.Model]);axes[0].set(ylim=(0,1.1),ylabel='Event sensitivity',title='Observed onset detection (always-alert = 1)')
        for x,(_,r) in enumerate(subset.iterrows()):axes[0].text(x,.7,f'{r.Detected}/{r.Eligible_Events}',ha='center')
        for i,name in enumerate(COLORS):
            q=m[(m.Model==name)&(m.Evaluation_Status=='eligible')];axes[1].scatter(np.full(len(q),i),q.Lead_Time_s,color=COLORS[name],s=20)
            severe=q[q.Has_91_92];axes[1].scatter(np.full(len(severe),i),severe.Lead_Time_s,facecolors='none',edgecolors='black',s=65)
        axes[1].set(xticks=range(4),xticklabels=list(COLORS),ylabel='First alert lead time (s)',ylim=(0,config.HORIZON+1),title='Eligible events; black ring = 91/92 episode')
        self.save(fig,'events_and_lead_time','Opportunity-based event matching is defined in EVALUATION.md. Nearby episodes can share an alert; high sensitivity under near-continuous alerts is not useful event precision. Severe subset is small.',m)
        if not (self.output/'latency_samples.csv.gz').exists():
            raise FileNotFoundError('Raw latency samples missing: run benchmark_latency before plotting distributions')
        lat=pd.read_csv(self.output/'latency_benchmark.csv');samples=pd.read_csv(self.output/'latency_samples.csv.gz')
        fig,axes=plt.subplots(1,2,figsize=(13,5))
        for name in COLORS:
            x=np.sort(samples[(samples.Model==name)&(samples.Component=='pulse_to_alert')].Latency_ms.to_numpy())
            axes[0].plot(x,np.arange(1,len(x)+1)/len(x),label=name,color=COLORS[name])
        axes[0].set(xscale='log',xlabel='Pulse-to-alert computational latency (ms, log scale)',ylabel='Empirical cumulative fraction',ylim=(0,1),title='Measured single-pulse latency distribution');axes[0].legend()
        components=['feature_calculation','model_only','thresholding','pulse_to_alert','model_loading']
        for i,name in enumerate(COLORS):
            q=lat[lat.Model==name].set_index('Component').loc[components]
            axes[1].plot(np.arange(len(q)),q.Median_ms,'o-',color=COLORS[name],label=name+' median')
            axes[1].scatter(np.arange(len(q)),q.P95_ms,marker='x',color=COLORS[name])
        axes[1].set(xticks=np.arange(5),xticklabels=['Features','Model only','Threshold','Pulse to alert','Load'],yscale='log',ylabel='Measured latency (ms, log scale)',title='Median (circle) and P95 (cross), independent timers');axes[1].legend(fontsize=8)
        self.save(fig,'latency_distributions','Warm single-thread empirical distributions; loading is separate. Log axes are explicit. Compute time excludes acquisition/monitor delay and the 30s prediction horizon; local peak confirmation needs another 8ms sample.',lat)

    def permutation(self,repeats=3):
        dest=self.directory/'block_permutation.csv';rows=[]
        with threadpool_limits(limits=1):
            for record,g in self.d.groupby('Record'):
                if g.Label.nunique()!=2:continue
                train=self.d[self.d.Record!=record];X=g[config.FEATURES].to_numpy(float);y=g.Label.to_numpy(int)
                for name in COLORS:
                    fitted=FittedModel(name).fit(train[config.FEATURES],train.Label)
                    score=fitted.score(X)
                    expected=self.p[(self.p.Model==name)&(self.p.Record==record)].sort_values('sample_id')
                    if not np.allclose(score,expected.Score,rtol=1e-8,atol=1e-8):raise ValueError('Diagnostic refit differs from evaluated held-out predictions')
                    baseline=average_precision_score(y,score)
                    for j,feature in enumerate(config.FEATURES):
                        for repeat in range(repeats):
                            rng=np.random.default_rng(config.SEED+repeat)
                            blocks=[np.arange(k,min(k+10,len(X))) for k in range(0,len(X),10)]
                            order=np.concatenate([blocks[k] for k in rng.permutation(len(blocks))]);changed=X.copy();changed[:,j]=X[order,j]
                            ap=average_precision_score(y,fitted.score(changed))
                            rows.append({'Model':name,'Record':record,'Feature':feature,'Repeat':repeat,'Baseline_AP':baseline,'Permuted_AP':ap,'AP_Drop':baseline-ap})
        values=pd.DataFrame(rows);values.to_csv(dest,index=False)
        fig,axes=plt.subplots(1,4,figsize=(15,6))
        for ax,name in zip(axes,COLORS):
            q=values[values.Model==name].groupby('Feature').AP_Drop.mean().reindex(config.FEATURES)
            ax.barh(q.index,q.values,color=COLORS[name]);ax.axvline(0,color='black',lw=.7);ax.set(xlabel='Mean held-out AP drop',title=name+' — diagnostic only')
        self.save(fig,'block_permutation_importance','Diagnostic fixed-model refits exclude each tested recording and reproduce its saved scores. Permute 10-pulse blocks, three fixed seeds, each feature separately, average only mixed-class recordings. Correlated features, implausible rate/interval combinations and few recordings limit interpretation. No features or models are selected or retuned; this is not conventional ELM weight importance.',values)

    def finish(self):
        sources=['run_manifest.json','causal_dataset.csv','per_record_predictions.csv','model_comparison.csv','dataset_audit.csv','per_recording_metrics.csv','event_metrics.csv','latency_benchmark.csv','latency_samples.csv.gz','threshold_analysis.csv.gz']
        receipt={'target':self.target,'run_manifest_sha256':fingerprint(self.output/'run_manifest.json'),
                 'sources':{name:fingerprint(self.output/name) for name in sources},
                 'raw_sources':self.manifest['source_hashes'],
                 'figure_code_sha256':fingerprint(Path(__file__)),
                 'files':{p.name:fingerprint(p) for p in sorted(self.directory.iterdir()) if p.is_file() and p.suffix in ['.png','.pdf','.csv']}}
        (self.directory/'figure_manifest.json').write_text(json.dumps(receipt,indent=2))
        text='# Figure index: '+self.target+'\n\nAll figures are generated from this run; source and image hashes are in figure_manifest.json. PNG previews and vector PDFs share each filename stem. Numerical panel tables are saved where relevant. No tuning was performed from these diagnostics.\n\n'
        for entry in self.index:
            name=entry['filename'];text+=f'- [{name}.png]({name}.png) / [PDF]({name}.pdf): {entry["interpretation"]}\n'
        (self.directory/'INDEX.md').write_text(text)
        return self.index


def generate_figures(output,importance=True):
    run=FigureRun(output)
    run.model_plots();run.dataset_plots();run.features();run.threshold_plots()
    for record in [12,26,30]:run.temporal(record)
    run.examples();run.events_latency()
    if importance:run.permutation()
    return run.finish()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--skip-importance',action='store_true');args=parser.parse_args()
    print(f'Generated {len(generate_figures(args.output,not args.skip_importance))} figures in {args.output}/figures')

if __name__=='__main__':main()
