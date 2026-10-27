"""Retrospective raw-signal/outcome/score panels; never fits or alters a model."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .audit import HERE,ROOT,SOURCE,sha

def main():
    failure=pd.read_csv(HERE/'event_failure_analysis.csv')
    pred=pd.read_csv(SOURCE/'classification_predictions.csv.gz',float_precision='round_trip')
    pred=pred[pred.Model.eq('Logistic')]
    events=failure[failure.Modality.eq('Combined')&failure.Model.eq('Logistic')&failure.Definition.eq('original')&failure.Qualifying_Onset].drop_duplicates(['Record','Event_ID'])
    folder=HERE/'figures/event_windows';folder.mkdir(exist_ok=True);cache={};index=[]
    colors={'SpO2_history':'#2563eb','PPG':'#7c3aed','Combined':'#0f766e'}
    for e in events.itertuples():
        r=int(e.Record);o=e.Onset_s
        if r not in cache:
            n=pd.read_csv(ROOT/f'data/bidmc_{r:02d}_Numerics.csv');n.columns=n.columns.str.strip()
            w=pd.read_csv(ROOT/f'data/bidmc_{r:02d}_Signals.csv');w.columns=w.columns.str.strip()
            cache[r]=(n,np.arange(len(w))/125+w['Time [s]'].iloc[0],w.PLETH.to_numpy())
        n,clock,ppg=cache[r];fig,axes=plt.subplots(3,1,figsize=(11,7),sharex=True,layout='constrained')
        use=n['Time [s]'].between(max(0,o-30),o+5);axes[0].step(n.loc[use,'Time [s]']-o,n.loc[use,'SpO2'],where='post',color='#1e293b');axes[0].axhline(94,color='red',linestyle='--');axes[0].set(ylabel='Observed SpO2 (%)')
        mask=(clock>=o-30)&(clock<=o+5);axes[1].plot(clock[mask]-o,ppg[mask],linewidth=.6,color='#475569');axes[1].set(ylabel='Raw PPG (export units)')
        for modality,color in colors.items():
            g=pred[pred.Record.eq(r)&pred.Modality.eq(modality)&pred.time_s.ge(o-30)&pred.time_s.lt(o)&pred.SpO2_Current.gt(94)]
            axes[2].scatter(g.time_s-o,g.Score-g.Threshold,label=modality,color=color,s=9)
            q=failure[failure.Record.eq(r)&failure.Event_ID.eq(e.Event_ID)&failure.Modality.eq(modality)&failure.Model.eq('Logistic')&failure.Definition.eq('original')].iloc[0]
            if q.Detected==True:axes[2].scatter([q.First_Alert_s-o],[0],marker='v',s=60,color=color)
        axes[2].axhline(0,color='black',linestyle='--');axes[2].set(ylabel='Score minus trained cutoff',xlabel='Seconds relative to observed onset');axes[2].legend(fontsize=8)
        for ax in axes:
            ax.axvline(0,color='red',linewidth=1);ax.axvspan(0,5,color='gray',alpha=.10);ax.set_xlim(-30,5)
        fig.suptitle(f'Record{r:02d}, event{e.Event_ID}; min{e.Minimum_SpO2:g}%, low span{e.Observed_Low_Span_s:g}s\n{e.Evaluation_Status}; raw traces are not validated SQI; triangles mark matched starts',fontsize=11)
        name=f'record{r:02d}_event{e.Event_ID}'
        for ext in ['png','pdf']:fig.savefig(folder/(name+'.'+ext),dpi=155,bbox_inches='tight')
        plt.close(fig);index.append(dict(Record=r,Event_ID=e.Event_ID,Onset_s=o,Original_Eligibility=e.Evaluation_Status,PNG=str((folder/(name+'.png')).relative_to(HERE)),PDF=str((folder/(name+'.pdf')).relative_to(HERE))))
    pd.DataFrame(index).to_csv(HERE/'event_window_index.csv',index=False)
    (HERE/'event_window_receipt.json').write_text(json.dumps(dict(Source_SHA256=sha(__file__),Events=len(index),Files={str(p.relative_to(HERE)):sha(p) for p in folder.iterdir() if p.is_file()},Limit='Retrospective visual inspection; no reference-annotated SQI, physiological cause or new performance claim'),indent=2))
    print('Exported',len(index),'qualifying-event panels, including the excluded22s onset')

if __name__=='__main__':main()
