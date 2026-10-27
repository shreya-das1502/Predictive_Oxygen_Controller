"""Read-only reanalysis of frozen held-out predictions; never trains or retunes."""
from pathlib import Path
import sys,json,hashlib,platform
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(ROOT))
from experiments.incremental_value_20261027.study import load_numeric,evaluate_record,topology,event_definition
from experiments.incremental_value_20261027.verify import stream_counts
from src.reliability_evaluation import episodes
OLD=ROOT/'experiments/incremental_value_20261027';OUT=OLD/'outputs'
MODELS=['Logistic','GaussianNB','ELM'];MODS=['SpO2_history','Combined'];LEADS=[0,5,10,15,20,30];SEED=20261027;B=2000

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def rate(a,b):return a/b if b else np.nan
def ci(values):
 v=np.asarray(values,float);v=v[np.isfinite(v)];return (*np.quantile(v,[.025,.975]),len(v)) if len(v) else (np.nan,np.nan,0)
def paired_bootstrap(deltas,records,weights):
 d=np.asarray(deltas,float);records=np.asarray(records,int);mean=np.full(len(weights),np.nan);median=mean.copy()
 for i,w in enumerate(weights):
  vals=np.repeat(d,w[records-1])
  if len(vals):mean[i]=vals.mean();median[i]=np.median(vals)
 return mean,median

def main():
 global HERE
 import argparse
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,help='NEW experiment directory for a reproducibility run');args=parser.parse_args()
 if args.out:
  target=args.out.resolve()
  if target.exists():raise ValueError('Refusing to overwrite an existing output directory')
  target.mkdir(parents=True);HERE=target
 if (HERE/'event_level_lead_times.csv').exists():raise ValueError('Results already exist: use --out with a NEW directory')
 sources=[OLD/'config.json',OLD/'study.py',OLD/'run.py',OLD/'analysis.py',OLD/'verify.py',ROOT/'src/config.py',ROOT/'src/preprocessing.py',ROOT/'src/features.py',ROOT/'src/labeling.py',ROOT/'src/reliability_evaluation.py',ROOT/'src/alarm_budget_experiment.py']
 sources += [OUT/n for n in ['classification_predictions.csv.gz','matched_dataset.csv.gz','raw_event_audit.csv','event_matches.csv','event_per_recording.csv','event_summary.csv','fold_ledger.json','selected_thresholds.csv','classification_fit_receipt.json','prepared_receipt.json','protocol.json','verification.json','independent_event_verification.json']]
 audit=ROOT/'experiments/final_audit_20261027T215012_IST';sources += [audit/n for n in ['REPORT.md','config.json','reproduced_metrics.json','paired_common_event_leads.csv','lead_coverage_diagnostic.csv','audit.py','engine.py']]
 sources += [ROOT/f'data/bidmc_{r:02d}_Numerics.csv' for r in range(1,31)]
 missing=[str(p.relative_to(ROOT)) for p in sources if not p.is_file()]
 if missing:dump(HERE/'blocked.json',dict(Missing=missing));raise FileNotFoundError(missing)
 # Reject altered prediction/preparation artifacts against their historical receipts.
 for receipt in ['classification_fit_receipt.json','prepared_receipt.json']:
  for n,h in json.loads((OUT/receipt).read_text()).items():
   if sha(OUT/n)!=h:raise ValueError('Frozen source receipt mismatch: '+n)
 pred=pd.read_csv(OUT/'classification_predictions.csv.gz',float_precision='round_trip');data=pd.read_csv(OUT/'matched_dataset.csv.gz',float_precision='round_trip');events=pd.read_csv(OUT/'raw_event_audit.csv',float_precision='round_trip');numeric=load_numeric()
 pred=pred[pred.Modality.isin(MODS)&pred.Model.isin(MODELS)].copy();assert len(events)==22 and events.Qualifying_Onset.sum()==15
 assert set(pred.Record)==set(range(1,31));assert data.sample_id.is_unique
 assert (data.peak_time_s<=data.time_s).all();np.testing.assert_allclose(data.time_s-data.peak_time_s,.008,atol=1e-10,rtol=0)
 np.testing.assert_array_equal(pred.Prediction,(pred.Score>=pred.Threshold).astype(int))
 ledger=json.loads((OUT/'fold_ledger.json').read_text());checked=0
 for f in ledger:
  if f['Modality'] not in MODS or f['Model'] not in MODELS:continue
  held=f['Outer_Test_Record'];assert set(f['Train_Records'])==set(range(1,31))-{held};assert f['Threshold_Selected_Before_Heldout_Scoring']
  for fold in f['Inner']:
   assert not set(fold['train_records'])&set(fold['validation_records']);assert held not in fold['train_records']+fold['validation_records']
  ck=OUT/'classification_folds'/f"{f['Modality']}__{f['Model']}__{held:02d}"
  receipt=ck/'complete.json';sources.append(receipt)
  for n,h in json.loads(receipt.read_text())['files'].items():
   p=ck/n;assert sha(p)==h;sources.append(p)
  grid=pd.read_csv(ck/'training_candidates.csv',float_precision='round_trip');winner=grid[grid.False_Alarm_Starts_Per_Hour.le(5+1e-12)&grid.Alarm_Active_Fraction.le(.05+1e-12)].sort_values(['Detected','Off_Event_Alarm_Starts','Alarm_Active_s','Threshold'],ascending=[False,True,True,False],kind='stable').iloc[0]
  assert winner.Threshold_hex==f['Selected']['Threshold_hex'];checked+=1
 oldper=pd.read_csv(OUT/'event_per_recording.csv',float_precision='round_trip');oldmatch=pd.read_csv(OUT/'event_matches.csv',float_precision='round_trip');allper=[];allmatches=[];alarms=[];verification=[]
 for (mod,model),g in pred.groupby(['Modality','Model']):
  assert len(g)==len(data) and g.sample_id.is_unique and set(g.sample_id)==set(data.sample_id)
  aligned=data.set_index('sample_id').loc[g.sample_id]
  for c in ['time_s','Record','Label','SpO2_Current']:np.testing.assert_array_equal(g[c],aligned[c])
  independent=stream_counts(g,events,numeric)
  for r,frame in g.groupby('Record'):
   normal,previous,duration=topology(frame,numeric);eps=episodes(normal,previous[1:])
   # Actual alarm starts only. Do not clip/restart runs at any event boundary.
   for a in eps:alarms.append(dict(Modality=mod,Model=model,Record=int(r),Alarm_ID=a['Alarm_ID'],Start_s=a['Start_s'],Last_Positive_s=a['End_s'],Positive_Timestamps=len(a['Positive_Times'])))
   for definition in ['original','sustained3']:
    counts,match=evaluate_record(frame,events,numeric,definition);counts.update(Modality=mod,Model=model);allper.append(counts)
    if len(match):allmatches.append(match.assign(Modality=mod,Model=model,Definition=definition))
    saved=oldper[oldper.Modality.eq(mod)&oldper.Model.eq(model)&oldper.Record.eq(r)&oldper.Definition.eq(definition)].iloc[0]
    for c in ['Eligible_Events','Detected','Missed_Events','Exposure_s','Alarm_Active_s','Off_Event_Alarm_Starts','Alarm_Starts','Unmatched_Alarm_Starts']:np.testing.assert_allclose(counts[c],saved[c],rtol=0,atol=1e-9)
  for definition,ind in independent.items():
   verification.append(dict(Modality=mod,Model=model,Definition=definition,IndependentDetected=ind['Detected'],IndependentEligible=ind['Eligible_Events']))
   hits={(r,e):(t,l) for r,e,t,l in ind['Matches']}
   selected=oldmatch[oldmatch.Modality.eq(mod)&oldmatch.Model.eq(model)&oldmatch.Definition.eq(definition)&oldmatch.Detected.eq(True)]
   assert set(hits)==set(zip(selected.Record,selected.Event_ID))
   for row in selected.itertuples():np.testing.assert_allclose(hits[(row.Record,row.Event_ID)],[row.First_Alert_s,row.Lead_Time_s],rtol=0,atol=1e-10)
 per=pd.DataFrame(allper);matches=pd.concat(allmatches,ignore_index=True);alarm=pd.DataFrame(alarms)
 rows=[]
 for model in MODELS:
  for definition in ['original','sustained3']:
   tables={mod:matches[matches.Modality.eq(mod)&matches.Model.eq(model)&matches.Definition.eq(definition)].set_index(['Record','Event_ID']) for mod in MODS}
   for event in events.to_dict('records'):
    key=(event['Record'],event['Event_ID']);v={k:event[k] for k in ['Record','Event_ID','Onset_s','End_s','Qualifying_Onset','Sustained3','Isolated_Reading']};v.update(Model=model,Definition=definition,Horizon_s=30.,WarningRule='earliest_unused_episode_start_in_[onset-30,onset)',TimelyRequirement_s=10.)
    for mod,prefix in [('SpO2_history','History'),('Combined','Combined')]:
     table=tables[mod]
     if key not in table.index:status='not_in_sustained_definition';detected=None;t=lead=np.nan
     else:
      q=table.loc[key];status=q.Evaluation_Status;detected=bool(q.Detected) if status=='eligible' else None;t=float(q.First_Alert_s);lead=float(q.Lead_Time_s)
      if status=='eligible':status='pre_onset_warning' if detected else 'no_qualifying_pre_onset_warning'
     v.update({prefix+'_EvaluationStatus':status,prefix+'_Detected':detected,prefix+'_Warning_s':t,prefix+'_Lead_s':lead,prefix+'_TimingClass':'before_onset' if detected else ('absent_under_rule' if detected is False else 'not_evaluable'),prefix+'_Timely10':bool(lead>=10) if detected is not None else None,prefix+'_PostOnsetWarningStatus':'not_assessed_by_prespecified_preonset_rule'})
    both=v['History_Detected'] is True and v['Combined_Detected'] is True
    v['Additional_Lead_s']=v['Combined_Lead_s']-v['History_Lead_s'] if both else np.nan;rows.append(v)
 event_table=pd.DataFrame(rows);event_table.to_csv(HERE/'event_level_lead_times.csv',index=False);alarm.to_csv(HERE/'alarm_episodes.csv',index=False)
 # Annotate all 30 recordings, including zero-event and zero-exposure recordings.
 leadg=event_table.groupby(['Model','Definition','Record'])
 for prefix,mod in [('History','SpO2_history'),('Combined','Combined')]:
  lut=leadg[prefix+'_Timely10'].sum(min_count=1)
  per.loc[per.Modality.eq(mod),'Timely10']=per[per.Modality.eq(mod)].apply(lambda q:lut.get((q.Model,q.Definition,q.Record),0),axis=1).fillna(0)
 per['Timely10']=per.Timely10.map(int).astype(int)
 per['InsufficientLead10_Detected']=per.Detected-per.Timely10;per['MissedOrInsufficient10']=per.Eligible_Events-per.Timely10
 per['Sensitivity']=per.Detected/per.Eligible_Events.replace(0,np.nan);per['Timely10Rate']=per.Timely10/per.Eligible_Events.replace(0,np.nan);per['MissedEventRate']=per.Missed_Events/per.Eligible_Events.replace(0,np.nan)
 per['FalseAlarmStartsPerEvaluableHour']=3600*per.Off_Event_Alarm_Starts/per.Exposure_s.replace(0,np.nan);per['AlarmActiveFraction']=per.Alarm_Active_s/per.Exposure_s.replace(0,np.nan)
 per['StartBudgetExceeded']=per.FalseAlarmStartsPerEvaluableHour.gt(5+1e-12);per['ActiveCapExceeded']=per.AlarmActiveFraction.gt(.05+1e-12);per['BurdenEvaluable']=per.Exposure_s.gt(0)
 per.to_csv(HERE/'recording_level_summary.csv',index=False)
 rng=np.random.default_rng(SEED);draw=rng.integers(0,30,size=(B,30));weights=np.array([np.bincount(d,minlength=30) for d in draw]);np.save(HERE/'recording_bootstrap_weights.npy',weights)
 summaries=[];pairs=[];intervals=[];timely=[];differences=[]
 for model in MODELS:
  for definition in ['original','sustained3']:
   ev=event_table[event_table.Model.eq(model)&event_table.Definition.eq(definition)];eligible=ev[ev.History_Detected.notna()];paired=eligible[eligible.History_Detected.eq(True)&eligible.Combined_Detected.eq(True)];delta=paired.Additional_Lead_s.to_numpy();mn,md=paired_bootstrap(delta,paired.Record.to_numpy(),weights)
   lo,hi,n=ci(md);ml,mh,mnvalid=ci(mn)
   pairs.append(dict(Model=model,Definition=definition,EligibleEvents=len(eligible),EligibleRecordings=eligible.Record.nunique(),PairedEvents=len(paired),PairedRecordings=paired.Record.nunique(),Earlier=int((delta>1e-9).sum()),Equal=int((np.abs(delta)<=1e-9).sum()),Later=int((delta<-1e-9).sum()),MedianAdditionalLead_s=float(np.median(delta)) if len(delta) else np.nan,MedianLower95_s=lo,MedianUpper95_s=hi,MedianDefinedBootstrapReplicates=n,MeanAdditionalLead_s=float(np.mean(delta)) if len(delta) else np.nan,MeanLower95_s=ml,MeanUpper95_s=mh,MeanDefinedBootstrapReplicates=mnvalid,BootstrapReplicates=B,BootstrapSeed=SEED,SamplingUnit='recording',ConditionalSubset='both_models_detected'))
   boot={}
   for prefix,mod in [('History','SpO2_history'),('Combined','Combined')]:
    table=per[per.Model.eq(model)&per.Definition.eq(definition)&per.Modality.eq(mod)].set_index('Record').reindex(range(1,31));tot={c:float(table[c].sum()) for c in ['Eligible_Events','Detected','Missed_Events','Timely10','Exposure_s','Alarm_Active_s','Off_Event_Alarm_Starts','Alarm_Starts','Unmatched_Alarm_Starts']}
    summary=dict(Model=model,Definition=definition,Modality=mod,**tot,EventBearingRecordings=int(table.Eligible_Events.gt(0).sum()),DetectedRecordings=int(table.Detected.gt(0).sum()),Timely10Recordings=int(table.Timely10.gt(0).sum()),ExposedRecordings=int(table.Exposure_s.gt(0).sum()),DetectionRate=rate(tot['Detected'],tot['Eligible_Events']),MissedEventRate=rate(tot['Missed_Events'],tot['Eligible_Events']),Timely10Rate=rate(tot['Timely10'],tot['Eligible_Events']),InsufficientLead10_Detected=tot['Detected']-tot['Timely10'],MissedOrInsufficient10=tot['Eligible_Events']-tot['Timely10'],FalseAlarmStartsPerEvaluableHour=rate(3600*tot['Off_Event_Alarm_Starts'],tot['Exposure_s']),AlarmActiveFraction=rate(tot['Alarm_Active_s'],tot['Exposure_s']),RecordStartBudgetViolations=int(table.StartBudgetExceeded.sum()),RecordActiveCapViolations=int(table.ActiveCapExceeded.sum()))
    summaries.append(summary);den=weights@table.Eligible_Events.to_numpy();boot[mod]={}
    for threshold in LEADS:
     t=eligible[prefix+'_Lead_s'].ge(threshold);counts=t.groupby(eligible.Record).sum().reindex(range(1,31),fill_value=0).to_numpy();num=weights@counts;vals=np.divide(num,den,out=np.full(B,np.nan),where=den>0);lo,hi,n=ci(vals);boot[mod][threshold]=vals
     timely.append(dict(Model=model,Definition=definition,Modality=mod,MinimumLead_s=threshold,Events=int(t.sum()),EligibleEvents=len(eligible),Rate=rate(t.sum(),len(eligible)),Lower95=lo,Upper95=hi,DefinedBootstrapReplicates=n,EventBearingRecordings=eligible.Record.nunique()))
   for threshold in LEADS:
    vals=boot['Combined'][threshold]-boot['SpO2_history'][threshold];lo,hi,n=ci(vals);a=next(t for t in timely if t['Model']==model and t['Definition']==definition and t['Modality']=='Combined' and t['MinimumLead_s']==threshold);b=next(t for t in timely if t['Model']==model and t['Definition']==definition and t['Modality']=='SpO2_history' and t['MinimumLead_s']==threshold)
    differences.append(dict(Model=model,Definition=definition,MinimumLead_s=threshold,CombinedMinusHistoryEvents=a['Events']-b['Events'],CombinedMinusHistoryRate=a['Rate']-b['Rate'],Lower95=lo,Upper95=hi,DefinedBootstrapReplicates=n))
 summary=pd.DataFrame(summaries);pair=pd.DataFrame(pairs);timely=pd.DataFrame(timely);diff=pd.DataFrame(differences)
 for n,table in [('overall_summary.csv',summary),('paired_lead_time_summary.csv',pair),('timely_warning_summary.csv',timely),('paired_warning_rate_differences.csv',diff)]:table.to_csv(HERE/n,index=False)
 # Reconcile every prespecified lead threshold with the preceding independent audit.
 oldleads=pd.read_csv(audit/'lead_coverage_diagnostic.csv')
 for q in timely.itertuples():
  v=oldleads[oldleads.Model.eq(q.Model)&oldleads.Definition.eq(q.Definition)&oldleads.Modality.eq(q.Modality)&oldleads.Minimum_Observed_Lead_s.eq(q.MinimumLead_s)].iloc[0];assert q.Events==v.Matched_Events_Meeting_Lead and q.EligibleEvents==v.Eligible_Events
 for definition,n in [('original',14),('sustained3',13)]:
  assert (summary[summary.Definition.eq(definition)].Eligible_Events==n).all()
 validate_tables(event_table,per,pair)
 plots(event_table,timely,summary)
 dump(HERE/'verification.json',dict(Status='passed',RecomputedConditions=12,IndependentStreamComparisons=verification,PriorLeadThresholdCounts='all_72_cells_identical',OuterFoldsChecked=checked,CausalFeatureClock='peak precedes confirmation by 8 ms; SpO2 at/before confirmation; prior causal implementation audit hash-sealed',PostOnsetWarnings='not measurable by preserved current-normal preonset matching rule',NoRefits=True,NoThresholdChanges=True,OutputSchemaAndArithmetic='passed'))
 sources=list(dict.fromkeys(sources));dump(HERE/'source_hashes.json',{str(p.relative_to(ROOT)):sha(p) for p in sources})
 print(summary.to_string(index=False));print('\nPAIRED\n'+pair.to_string(index=False))

def validate_tables(events,per,pair):
 assert not events.duplicated(['Model','Definition','Record','Event_ID']).any();assert len(events)==132
 for prefix in ['History','Combined']:
  hits=events[prefix+'_Detected'].eq(True);miss=events[prefix+'_Detected'].eq(False)
  assert events.loc[miss,prefix+'_Lead_s'].isna().all() and events.loc[miss,prefix+'_Warning_s'].isna().all()
  np.testing.assert_allclose(events.loc[hits,'Onset_s']-events.loc[hits,prefix+'_Warning_s'],events.loc[hits,prefix+'_Lead_s'],atol=1e-10,rtol=0)
  assert events.loc[hits,prefix+'_Lead_s'].gt(0).all() and events.loc[hits,prefix+'_Lead_s'].le(30).all()
  for key,g in events[hits].groupby(['Model','Definition','Record']):assert not g[prefix+'_Warning_s'].duplicated().any()
 both=events.History_Detected.eq(True)&events.Combined_Detected.eq(True);assert events.loc[~both,'Additional_Lead_s'].isna().all()
 np.testing.assert_allclose(events.loc[both,'Additional_Lead_s'],events.loc[both,'Combined_Lead_s']-events.loc[both,'History_Lead_s'])
 assert (per.Detected+per.Missed_Events==per.Eligible_Events).all();assert (per.Timely10<=per.Detected).all()
 assert (pair.Earlier+pair.Equal+pair.Later==pair.PairedEvents).all()

def plots(events,timely,summary):
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':220})
 colors=['#2563a6','#d36b24'];labels=['SpO₂ history','SpO₂ + PPG']
 fig,axes=plt.subplots(3,2,figsize=(12,13),layout='constrained')
 for i,model in enumerate(MODELS):
  for j,definition in enumerate(['original','sustained3']):
   ax=axes[i,j];ev=events[events.Model.eq(model)&events.Definition.eq(definition)&events.History_Detected.notna()];x=np.arange(len(ev))
   for k,prefix in enumerate(['History','Combined']):
    lead=ev[prefix+'_Lead_s'].to_numpy(float);good=np.isfinite(lead);xx=x+(k-.5)*.22;ax.scatter(xx[good],lead[good],s=33,color=colors[k],label=labels[k]);ax.scatter(xx[~good],np.full((~good).sum(),-4),marker='x',color=colors[k],s=38)
   ax.axhline(10,color='#777',ls='--',lw=.8);ax.axhline(-1,color='#aaa',lw=.6);ax.set_ylim(-7,32);ax.set_yticks([-4,0,10,20,30],['Missed','0','10','20','30']);ax.set_xticks(x,[f'{r.Record}:{r.Event_ID}' for r in ev.itertuples()],rotation=65,ha='right',fontsize=8);ax.set(title=f'{model} · {definition} (N={len(ev)})',ylabel='Pre-onset warning lead (s)',xlabel='Recording:event');ax.legend(fontsize=8,loc='upper right')
 fig.suptitle('Frozen-model warning times: misses shown separately, never assigned zero lead',fontsize=13);fig.savefig(HERE/'warning_lead_time_comparison.png');plt.close(fig)
 fig,axes=plt.subplots(3,2,figsize=(11,11),layout='constrained')
 for i,model in enumerate(MODELS):
  for j,definition in enumerate(['original','sustained3']):
   ax=axes[i,j]
   for k,mod in enumerate(MODS):
    t=timely[timely.Model.eq(model)&timely.Definition.eq(definition)&timely.Modality.eq(mod)].sort_values('MinimumLead_s');ax.plot(t.MinimumLead_s,t.Rate,marker='o',label=labels[k],color=colors[k]);ax.fill_between(t.MinimumLead_s,t.Lower95,t.Upper95,color=colors[k],alpha=.12)
   ax.axvline(10,color='#777',ls='--',lw=.8);ax.set(ylim=(-.03,1.03),xticks=LEADS,xlabel='Required pre-onset lead (s); 0 = detected',ylabel='Fraction of all eligible events',title=f'{model} · {definition}');ax.legend(fontsize=8)
 fig.suptitle('Timely coverage retains misses in denominator; 95% recording-cluster intervals',fontsize=13);fig.savefig(HERE/'timely_warning_comparison.png');plt.close(fig)

if __name__=='__main__':main()
