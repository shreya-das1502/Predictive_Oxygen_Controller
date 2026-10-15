"""Training-partition-only fitting and causal experimental context; no production edits."""
from collections import deque
import hashlib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from src import config
from src.model import FittedModel

CONTEXT = ['Amplitude_Mean_30s','Amplitude_CV_30s','Rate_Mean_30s','Rate_Trend_30s',
           'Interval_CV_30s','Amplitude_Ratio_30s','Recent_Pulse_Count_10s','Since_Previous_Pulse_s']

class ContextState:
    def __init__(self):self.history=deque();self.last=None
    def update(self,row):
        t=float(row['time_s'])
        if not np.isfinite(t) or (self.last is not None and t<=self.last):raise ValueError('Reset context at recording boundaries or backward time')
        previous=self.last;self.last=t
        self.history.append((t,float(row['PPG_Amplitude']),float(row['Pulse_Rate_BPM']),float(row['Pulse_Interval_s'])))
        while self.history[0][0]<t-30:self.history.popleft()
        h=np.asarray(self.history);a=h[:,1];rate=h[:,2];iv=h[:,3]
        def cv(v):return float(np.std(v,ddof=1)/np.mean(v)) if len(v)>1 and np.mean(v)>0 else 0.
        x=h[:,0]-h[:,0].mean();den=x@x
        slope=float(x@(rate-rate.mean())/den) if den>0 else 0.
        return np.array([a.mean(),cv(a),rate.mean(),slope,cv(iv),a[-1]/a.mean() if a.mean()>0 else np.nan,
                         np.sum(h[:,0]>=t-10),t-previous if previous is not None else 0.])


def add_context(raw):
    rows=[]
    for _,g in raw.groupby('Record',sort=False):
        state=ContextState()
        if (np.diff(g.time_s)<=0).any():raise ValueError('Pulse history must be chronological')
        for row in g.to_dict('records'):rows.append(dict(Record=row['Record'],pulse_row=row['pulse_row'],**dict(zip(CONTEXT,state.update(row)))))
    return pd.DataFrame(rows)


def fit_model(frame,features,name):
    # Same estimator and hyperparameters; isolated in-memory experimental schema.
    model=FittedModel(name);model.features=list(features)
    return model.fit(frame[features].to_numpy(float),frame.Label.to_numpy(int))


def negative_rank(reference, scores):
    reference=np.asarray(reference,float);scores=np.asarray(scores,float)
    if not len(reference) or not np.isfinite(reference).all() or not np.isfinite(scores).all():raise ValueError('Finite training-negative reference required')
    return np.searchsorted(np.sort(reference),scores,side='right')/len(reference)


def training_scores(train,features,name,outer_test):
    """Receives ONLY outer training rows. Calibration references never include validation."""
    if outer_test in set(train.Record):raise ValueError('Outer record leaked into fitting')
    raw=np.full(len(train),np.nan);rank=raw.copy();splits=[]
    for tr,va in GroupKFold(n_splits=3).split(train,train.Label,train.Record):
        fit=train.iloc[tr];validation=train.iloc[va]
        if set(fit.Record)&set(validation.Record):raise ValueError('Inner recording leakage')
        model=fit_model(fit,features,name)
        values=model.score(validation[features].to_numpy(float))
        reference=model.score(fit.loc[fit.Label.eq(0),features].to_numpy(float))
        raw[va]=values;rank[va]=negative_rank(reference,values)
        splits.append(dict(train_records=sorted(map(int,fit.Record.unique())),validation_records=sorted(map(int,validation.Record.unique()))))
    if not np.isfinite(raw).all():raise ValueError('Incomplete grouped OOF coverage')
    model=fit_model(train,features,name)
    reference=model.score(train.loc[train.Label.eq(0),features].to_numpy(float))
    return dict(raw=raw,negative_rank=rank),model,reference,splits


def array_hash(values):
    return hashlib.sha256(np.asarray(values,dtype='<f8').tobytes()).hexdigest()
