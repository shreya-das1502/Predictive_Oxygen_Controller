"""Matched held-out evaluation, conditional record bootstrap and exportable figures."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, brier_score_loss
from src.evaluation import metrics
from .study import evaluate_record, sha

KEYS=['Modality','Model']

def ratio(a,b):
    a,b=np.broadcast_arrays(np.asarray(a,float),np.asarray(b,float))
    return np.divide(a,b,out=np.full(a.shape,np.nan),where=b!=0)

def ci(values):
    v=np.asarray(values,float);v=v[np.isfinite(v)]
    q=np.quantile(v,[.025,.975]) if len(v) else [np.nan,np.nan]
    return dict(Lower_95=float(q[0]),Upper_95=float(q[1]),Defined_Replicates=len(v))

def conditional_bootstrap(per,weights):
    g=per.set_index('Record').reindex(range(1,31))
    for c in ['Detected','Eligible_Events','Off_Event_Alarm_Starts','Exposure_s','Alarm_Active_s']:
        if g[c].isna().any():raise AssertionError('Reporting roster incomplete')
    sums={c:weights@g[c].to_numpy(float) for c in ['Detected','Eligible_Events','Off_Event_Alarm_Starts','Exposure_s','Alarm_Active_s']}
    return dict(Sensitivity=ratio(sums['Detected'],sums['Eligible_Events']),False_Alarm_Starts_Per_Hour=ratio(3600*sums['Off_Event_Alarm_Starts'],sums['Exposure_s']),Alarm_Active_Fraction=ratio(sums['Alarm_Active_s'],sums['Exposure_s']))

def analyse(out,protocol,data,numeric,events):
    out=Path(out)
    for receipt in ['classification_fit_receipt.json','forecast_fit_receipt.json','prepared_receipt.json']:
        for n,want in json.loads((out/receipt).read_text()).items():
            if sha(out/n)!=want:raise ValueError('Input receipt mismatch: '+n)
    pred=pd.read_csv(out/'classification_predictions.csv.gz',float_precision='round_trip')
    forecast=pd.read_csv(out/'forecast_predictions.csv.gz',float_precision='round_trip')
    np.testing.assert_array_equal(pred.Prediction,(pred.Score>=pred.Threshold).astype(int))
    expected=set(data.sample_id);per=[];matches=[];timestamp=[];timestamp_record=[];calibration=[];probability=[]
    for (modality,model),full in pred.groupby(KEYS,sort=True):
        assert len(full)==len(data) and not full.sample_id.duplicated().any() and set(full.sample_id)==expected
        for population,g in [('full_target',full),('current_normal',full[full.SpO2_Current.gt(94)])]:
            timestamp.append(dict(Modality=modality,Model=model,Population=population,**metrics(g.Label,g.Score,g.Prediction)))
            for record in range(1,31):
                h=g[g.Record.eq(record)];timestamp_record.append(dict(Modality=modality,Model=model,Population=population,Record=record,**metrics(h.Label,h.Score,h.Prediction)))
            valid_probability=model in ['Logistic','GaussianNB']
            probability.append(dict(Modality=modality,Model=model,Population=population,Probability_Metric_Valid=valid_probability,Brier=brier_score_loss(g.Label,g.Score) if valid_probability else np.nan,Score_Semantics='uncalibrated probability-shaped classifier score' if valid_probability else 'not a probability; no Brier assessment'))
            if valid_probability:
                if not g.Score.between(0,1).all():raise AssertionError('Invalid probability-shaped score')
                bins=np.minimum((g.Score.to_numpy()*10).astype(int),9)
                for k in range(10):
                    h=g.iloc[np.flatnonzero(bins==k)]
                    calibration.append(dict(Modality=modality,Model=model,Population=population,Bin=k,Lower=k/10,Upper=(k+1)/10,N=len(h),Mean_Score=h.Score.mean(),Observed_Rate=h.Label.mean()))
        for record,g in full.groupby('Record',sort=True):
            g=g.sort_values('time_s').reset_index(drop=True)
            for definition in ['original','sustained3']:
                result,match=evaluate_record(g,events,numeric,definition)
                per.append(dict(Modality=modality,Model=model,**result))
                if len(match):matches.append(match.assign(Modality=modality,Model=model,Definition=definition,Policy=g.Policy.iloc[0]))
    per=pd.DataFrame(per);match=pd.concat(matches,ignore_index=True);ts=pd.DataFrame(timestamp)
    rng=np.random.default_rng(protocol['bootstrap_seed']);draws=rng.integers(0,30,size=(protocol['bootstrap_repetitions'],30))
    weights=np.array([np.bincount(d,minlength=30) for d in draws]);pd.DataFrame(weights,columns=[f'Record_{r:02d}' for r in range(1,31)]).to_csv(out/'bootstrap_recording_weights.csv.gz',index=False,compression='gzip')
    event_summary=[];intervals=[];boots={}
    countcols=['Detected','Eligible_Events','Missed_Events','Exposure_s','Alarm_Active_s','Off_Event_Alarm_Starts','Alarm_Starts','Unmatched_Alarm_Starts','Samples','Alert_Timestamps']
    for key,g in per.groupby(KEYS+['Definition'],sort=True):
        context=dict(zip(KEYS+['Definition'],key));v={c:float(g[c].sum()) for c in countcols}
        for c in ['Detected','Eligible_Events','Missed_Events','Off_Event_Alarm_Starts','Alarm_Starts','Unmatched_Alarm_Starts','Samples','Alert_Timestamps']:v[c]=int(v[c])
        leads=match[(match.Modality==key[0])&(match.Model==key[1])&(match.Definition==key[2])&match.Evaluation_Status.eq('eligible')&match.Detected.eq(True)].Lead_Time_s.dropna()
        v.update(Sensitivity=float(ratio(v['Detected'],v['Eligible_Events'])),False_Alarm_Starts_Per_Hour=float(ratio(3600*v['Off_Event_Alarm_Starts'],v['Exposure_s'])),Alarm_Active_Fraction=float(ratio(v['Alarm_Active_s'],v['Exposure_s'])),Lead_Median_s=leads.median(),Lead_Min_s=leads.min(),Lead_Max_s=leads.max(),Pooled_Research_Budget_Adherence=bool(3600*v['Off_Event_Alarm_Starts']/v['Exposure_s']<=5+1e-12 and v['Alarm_Active_s']/v['Exposure_s']<=.05+1e-12),Record_Budget_Violations=int(((g.Exposure_s>0)&(3600*g.Off_Event_Alarm_Starts/g.Exposure_s>5+1e-12)).sum()),Record_Active_Cap_Violations=int(((g.Exposure_s>0)&(g.Alarm_Active_s/g.Exposure_s>.05+1e-12)).sum()),Policy='fixed_not_budget_selected' if key[1] in ['Persistence','Trend10'] else 'training_OOF_budget5_cap005')
        event_summary.append(context|v);b=conditional_bootstrap(g,weights);boots[key]=b
        for metric,values in b.items():intervals.append(context|dict(Metric=metric,**ci(values),Unit='whole_recording',Refits=False))
    es=pd.DataFrame(event_summary)
    for definition,g in es.groupby('Definition'):
        assert g.Eligible_Events.nunique()==1 and np.isclose(g.Exposure_s,g.Exposure_s.iloc[0]).all()
    paired=[]
    for model in protocol['classifiers']:
        for modality in ['Combined','PPG']:
            for definition in ['original','sustained3']:
                a=es[(es.Modality==modality)&es.Model.eq(model)&es.Definition.eq(definition)].iloc[0]
                b=es[es.Modality.eq('SpO2_history')&es.Model.eq(model)&es.Definition.eq(definition)].iloc[0]
                for metric in ['Sensitivity','False_Alarm_Starts_Per_Hour','Alarm_Active_Fraction']:
                    values=boots[(modality,model,definition)][metric]-boots[('SpO2_history',model,definition)][metric]
                    paired.append(dict(Model=model,Contrast=modality+' minus SpO2_history',Definition=definition,Metric=metric,Difference=a[metric]-b[metric],**ci(values)))
    # Event audit retains isolated/excluded episodes even under the sensitivity definition.
    audit=[]
    columns=['Record','Event_ID','Evaluation_Status','Detected','First_Alert_s','Lead_Time_s','Alarm_ID','Opportunities','Policy']
    for (modality,model,definition),g in match.groupby(KEYS+['Definition']):
        d=events.merge(g[columns],on=['Record','Event_ID'],how='left',validate='one_to_one')
        d['Evaluation_Status']=d.Evaluation_Status.fillna('not_sustained_under_sensitivity_definition')
        audit.append(d.assign(Modality=modality,Model=model,Definition=definition))
    audit=pd.concat(audit,ignore_index=True)
    # Forecast errors use identical held-out rows and retain raw/clipped outputs.
    fper=[]
    for model,g in forecast.groupby('Model'):
        assert len(g)==len(data) and set(g.sample_id)==expected and not g.sample_id.duplicated().any()
        for population,part in [('full_target',g),('current_normal',g[g.SpO2_Current.gt(94)])]:
            for record in range(1,31):
                h=part[part.Record.eq(record)];err=h.Forecast-h.FutureMin
                fper.append(dict(Model=model,Population=population,Record=record,N=len(h),Absolute_Error_Sum=float(np.abs(err).sum()),Squared_Error_Sum=float((err**2).sum()),MAE=float(np.abs(err).mean()) if len(h) else np.nan,RMSE=float(np.sqrt((err**2).mean())) if len(h) else np.nan,Clipped_Rows=int((h.Raw_Forecast!=h.Forecast).sum())))
    fper=pd.DataFrame(fper);fsum=[];fboots={};fcis=[]
    for (model,population),g in fper.groupby(['Model','Population']):
        h=g.set_index('Record').reindex(range(1,31));n=h.N.sum();mae=h.Absolute_Error_Sum.sum()/n;rmse=np.sqrt(h.Squared_Error_Sum.sum()/n)
        fsum.append(dict(Model=model,Population=population,N=int(n),MAE=mae,RMSE=rmse,Clipped_Rows=int(h.Clipped_Rows.sum()),Unit='SpO2_percentage_points'))
        b=dict(MAE=ratio(weights@h.Absolute_Error_Sum.to_numpy(),weights@h.N.to_numpy()),RMSE=np.sqrt(ratio(weights@h.Squared_Error_Sum.to_numpy(),weights@h.N.to_numpy())))
        fboots[(model,population)]=b
        for metric,values in b.items():fcis.append(dict(Model=model,Population=population,Metric=metric,**ci(values),Unit='whole_recording'))
    fsum=pd.DataFrame(fsum);fpairs=[]
    for population in ['full_target','current_normal']:
        for reference in ['Ridge_SpO2_history','Persistence','Trend10']:
            for metric in ['MAE','RMSE']:
                a=fsum[fsum.Model.eq('Ridge_Combined')&fsum.Population.eq(population)].iloc[0];b=fsum[fsum.Model.eq(reference)&fsum.Population.eq(population)].iloc[0]
                fpairs.append(dict(Contrast='Ridge_Combined minus '+reference,Population=population,Metric=metric,Difference=a[metric]-b[metric],**ci(fboots[('Ridge_Combined',population)][metric]-fboots[(reference,population)][metric])))
    changes=[]
    for model in protocol['classifiers']:
        for definition in ['original','sustained3']:
            base=match[match.Model.eq(model)&match.Modality.eq('SpO2_history')&match.Definition.eq(definition)&match.Evaluation_Status.eq('eligible')]
            combined=match[match.Model.eq(model)&match.Modality.eq('Combined')&match.Definition.eq(definition)&match.Evaluation_Status.eq('eligible')]
            joined=base[['Record','Event_ID','Detected']].merge(combined[['Record','Event_ID','Detected']],on=['Record','Event_ID'],validate='one_to_one',suffixes=('_SpO2','_Combined'))
            for r in joined.itertuples():
                state='both_detect' if r.Detected_SpO2 and r.Detected_Combined else 'combined_gain' if r.Detected_Combined else 'combined_loss' if r.Detected_SpO2 else 'both_miss'
                changes.append(dict(Model=model,Definition=definition,Record=r.Record,Event_ID=r.Event_ID,SpO2_Detected=r.Detected_SpO2,Combined_Detected=r.Detected_Combined,Change=state))
    tables={'timestamp_summary.csv':ts,'timestamp_per_recording.csv':pd.DataFrame(timestamp_record),'calibration_bins.csv':pd.DataFrame(calibration),'probability_assessment.csv':pd.DataFrame(probability),'event_per_recording.csv':per,'event_matches.csv':match,'event_summary.csv':es,'event_audit_long.csv':audit,'event_bootstrap_intervals.csv':pd.DataFrame(intervals),'paired_modality_differences.csv':pd.DataFrame(paired),'paired_event_changes.csv':pd.DataFrame(changes),'forecast_per_recording.csv':fper,'forecast_summary.csv':fsum,'forecast_bootstrap_intervals.csv':pd.DataFrame(fcis),'paired_forecast_differences.csv':pd.DataFrame(fpairs)}
    ablation=ts.merge(es[es.Definition.eq('original')].drop(columns='Definition'),on=KEYS,suffixes=('_Timestamp','_Event'))
    tables['feature_modality_ablation.csv']=ablation
    for name,table in tables.items():table.to_csv(out/name,index=False)
    figures(out,pred,ts,es,match,forecast,fsum,pd.DataFrame(calibration))
    (out/'analysis_receipt.json').write_text(json.dumps(dict(analysis_source_sha256=sha(Path(__file__)),tables={n:sha(out/n) for n in tables},figures={p.name:sha(p) for p in (out/'figures').iterdir() if p.is_file()},conditional_bootstrap=True,clinical_validation=False),indent=2))
    print('ANALYSED',len(pred),'classification predictions;',len(forecast),'forecasts;',len(es),'event conditions',flush=True)

def figures(out,pred,ts,es,match,forecast,fsum,calibration):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    folder=out/'figures';folder.mkdir(exist_ok=True)
    colors={'SpO2_history':'#2563eb','PPG':'#7c3aed','Combined':'#0f766e'}
    def save(fig,name):
        for ext in ['png','pdf']:fig.savefig(folder/(name+'.'+ext),dpi=160,bbox_inches='tight')
        plt.close(fig)
    for population in ['full_target','current_normal']:
        fig,axes=plt.subplots(1,3,figsize=(13,4.5),layout='constrained')
        for ax,model in zip(axes,['Logistic','GaussianNB','ELM']):
            for modality in colors:
                g=pred[pred.Model.eq(model)&pred.Modality.eq(modality)]
                if population=='current_normal':g=g[g.SpO2_Current.gt(94)]
                precision,recall,_=precision_recall_curve(g.Label,g.Score)
                ax.plot(recall,precision,label=f'{modality} AP={average_precision_score(g.Label,g.Score):.3f}',color=colors[modality])
            ax.axhline(g.Label.mean(),color='gray',linestyle='--',linewidth=1);ax.set(title=model,xlabel='Timestamp recall',ylabel='Timestamp precision',xlim=(0,1),ylim=(0,1));ax.legend(fontsize=7)
        fig.suptitle('Held-out precision–recall: '+population+' (pooled cross-fit scores)');save(fig,'precision_recall_'+population)
    primary=es[es.Definition.eq('original')].copy();labels=primary.Modality+'/'+primary.Model;x=np.arange(len(primary))
    fig,axes=plt.subplots(3,1,figsize=(11,9),layout='constrained')
    for ax,col,title in zip(axes,['Sensitivity','False_Alarm_Starts_Per_Hour','Alarm_Active_Fraction'],['One-to-one eligible-event sensitivity','Off-event alarm starts / evaluable hour','Time-weighted alarm-active fraction']):
        ax.bar(x,primary[col],color=[colors[v] for v in primary.Modality]);ax.set(ylabel=title);ax.set_xticks(x,labels,rotation=35,ha='right',fontsize=8)
        if col=='Sensitivity':
            ax.set_ylim(0,1.12)
            for i,r in enumerate(primary.itertuples()):ax.text(i,r.Sensitivity+.015,f'{r.Detected}/{r.Eligible_Events}',ha='center',fontsize=8)
        elif col=='False_Alarm_Starts_Per_Hour':ax.axhline(5,color='red',linestyle='--',label='Research budget5');ax.legend()
        else:ax.axhline(.05,color='red',linestyle='--',label='Research cap5%');ax.legend()
    fig.suptitle('Primary event policy: learned OOF constraints; fixed references are not budget-selected');save(fig,'event_detection_and_burden')
    fig,ax=plt.subplots(figsize=(11,4.8),layout='constrained')
    for i,r in enumerate(primary.itertuples()):
        leads=match[match.Modality.eq(r.Modality)&match.Model.eq(r.Model)&match.Definition.eq('original')&match.Detected.eq(True)].Lead_Time_s.dropna()
        if len(leads):ax.scatter(np.full(len(leads),i),leads,s=22,color=colors[r.Modality])
        else:ax.text(i,1,'undefined',rotation=90,ha='center',va='bottom',fontsize=8)
    ax.set_xticks(x,labels,rotation=35,ha='right',fontsize=8);ax.set(ylabel='Matched-event lead (s)',ylim=(0,31),title='Lead times condition on detections; misses are not assigned zero');save(fig,'lead_times')
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for ax,population in zip(axes,['full_target','current_normal']):
        g=fsum[fsum.Population.eq(population)].sort_values('Model');ix=np.arange(len(g));ax.bar(ix-.18,g.MAE,.36,label='MAE');ax.bar(ix+.18,g.RMSE,.36,label='RMSE');ax.set_xticks(ix,g.Model,rotation=30,ha='right',fontsize=8);ax.set(title=population,ylabel='SpO2 percentage points');ax.legend()
    fig.suptitle('Held-out continuous future-minimum forecasting (not event-detection proof)');save(fig,'forecast_errors')
    fig,axes=plt.subplots(1,2,figsize=(10,4.5),layout='constrained')
    for ax,model in zip(axes,['Logistic','GaussianNB']):
        ax.plot([0,1],[0,1],color='gray',linestyle='--')
        for modality in colors:
            g=calibration[calibration.Model.eq(model)&calibration.Modality.eq(modality)&calibration.Population.eq('full_target')&calibration.N.gt(0)]
            ax.plot(g.Mean_Score,g.Observed_Rate,marker='o',label=modality,color=colors[modality])
        ax.set(title=model+' (uncalibrated)',xlabel='Mean model output',ylabel='Observed timestamp positive rate',xlim=(0,1),ylim=(0,1));ax.legend(fontsize=8)
    fig.suptitle('Reliability diagnostics: weighted/prior-shifted scores; no fitted calibration');save(fig,'calibration_diagnostics')
    e=match[match.Definition.eq('original')&match.Evaluation_Status.eq('eligible')].copy();e['Event']=e.Record.astype(str)+':'+e.Event_ID.astype(str);e['Condition']=e.Modality+'/'+e.Model
    matrix=e.pivot(index='Condition',columns='Event',values='Detected').astype(float)
    fig,ax=plt.subplots(figsize=(12,5),layout='constrained');ax.imshow(matrix.to_numpy(),vmin=0,vmax=1,cmap=ListedColormap(['#fee2e2','#0f766e']),aspect='auto');ax.set_xticks(np.arange(len(matrix.columns)),matrix.columns,rotation=60,ha='right',fontsize=8);ax.set_yticks(np.arange(len(matrix)),matrix.index,fontsize=8);ax.set(title='Primary event audit: red=missed, green=detected; record:event identifier');save(fig,'event_error_audit')
