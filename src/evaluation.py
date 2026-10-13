"""Recording-separated fitting, grouped OOF thresholding, and explicit metrics."""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.metrics import average_precision_score, roc_auc_score, matthews_corrcoef, precision_recall_curve
from .config import FEATURES, TARGET_RECALL, INNER_FOLDS
from .model import FittedModel


def outer_splits(groups):
    groups=np.asarray(groups)
    for train,test in LeaveOneGroupOut().split(np.zeros(len(groups)),groups=groups):
        if set(groups[train]) & set(groups[test]):raise AssertionError('Recording leakage')
        yield train,test


def threshold_table(y,scores):
    """One stable sort and cumulative counts; ties are evaluated as complete blocks."""
    y=np.asarray(y,int);s=np.asarray(scores,float)
    if not len(y) or len(y)!=len(s) or not np.isin(y,[0,1]).all() or not np.isfinite(s).all():raise ValueError('Threshold selection requires finite aligned scores and binary labels')
    order=np.argsort(-s,kind='stable');s=s[order];y=y[order]
    ends=np.r_[np.flatnonzero(np.diff(s)!=0),len(s)-1]
    tp=np.cumsum(y)[ends];fp=ends+1-tp;p=y.sum();n=len(y)-p
    return pd.DataFrame({'Threshold':s[ends],'TP':tp,'FP':fp,'FN':p-tp,'TN':n-fp,
                         'Recall':tp/p if p else np.full(len(tp),np.nan),
                         'Precision':tp/(tp+fp),'Specificity':1-fp/n if n else np.full(len(tp),np.nan),
                         'FPR':fp/n if n else np.full(len(tp),np.nan)})


def choose_threshold(y,scores):
    table=threshold_table(y,scores)
    if not np.any(np.asarray(y)==1):raise ValueError('No validation positives; target-recall threshold cannot be selected')
    eligible=table[table.Recall>=TARGET_RECALL]
    # Descending thresholds: first is largest, hence maximizes specificity at target recall.
    if eligible.empty:raise ValueError('Target recall unavailable; no silent target relaxation')
    return float(eligible.iloc[0].Threshold),table


def grouped_oof(X,y,groups,name):
    """Only receives outer TRAIN arrays: outer outcomes are inaccessible here."""
    X,y,groups=np.asarray(X),np.asarray(y),np.asarray(groups)
    n_groups=len(np.unique(groups))
    if n_groups<2:raise ValueError('At least two training recordings needed for grouped validation')
    scores=np.full(len(y),np.nan);folds=[]
    for train,val in GroupKFold(n_splits=min(INNER_FOLDS,n_groups)).split(X,y,groups):
        if set(groups[train]) & set(groups[val]):raise AssertionError('Inner recording leakage')
        if len(np.unique(y[train]))<2:raise ValueError(f'{name}: grouped inner training fold has one class; cannot fit fairly')
        fitted=FittedModel(name).fit(X[train],y[train]);scores[val]=fitted.score(X[val])
        folds.append({'train_records':sorted(map(int,np.unique(groups[train]))),'validation_records':sorted(map(int,np.unique(groups[val])))})
    if not np.isfinite(scores).all():raise AssertionError('Each training row must have one OOF prediction')
    threshold,table=choose_threshold(y,scores)
    return threshold,scores,table,folds


def metrics(y,scores,predictions):
    y=np.asarray(y,int);p=np.asarray(predictions,int);s=np.asarray(scores,float)
    tp=int(((y==1)&(p==1)).sum());fn=int(((y==1)&(p==0)).sum());fp=int(((y==0)&(p==1)).sum());tn=int(((y==0)&(p==0)).sum())
    recall=tp/(tp+fn) if tp+fn else np.nan;precision=tp/(tp+fp) if tp+fp else np.nan;specificity=tn/(tn+fp) if tn+fp else np.nan
    return {'Samples':len(y),'Positives':int(y.sum()),'Negatives':int((y==0).sum()),'TP':tp,'FN':fn,'FP':fp,'TN':tn,
            'Recall':recall,'Precision':precision,'Specificity':specificity,'FPR':1-specificity,
            'F1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else np.nan,
            'Balanced_Accuracy':(recall+specificity)/2 if np.isfinite(recall) and np.isfinite(specificity) else np.nan,
            'MCC':float(matthews_corrcoef(y,p)) if len(np.unique(y))==2 and len(np.unique(p))==2 else np.nan,
            'PR_AUC_AP':float(average_precision_score(y,s)) if len(np.unique(y))==2 else np.nan,
            'ROC_AUC':float(roc_auc_score(y,s)) if len(np.unique(y))==2 else np.nan,
            'Predicted_Positive':int(p.sum()),'Alert_Fraction':float(p.mean()) if len(p) else np.nan,'Target_Reached':bool(recall>=TARGET_RECALL) if np.isfinite(recall) else np.nan}


def alarm_statistics(group):
    """Alert episodes: positive runs, split at >2s gaps; not one alarm per pulse."""
    g=group.sort_values('time_s');t=g.time_s.to_numpy();p=g.Prediction.to_numpy(bool)
    indices=np.flatnonzero(p & np.r_[True,(~p[:-1])|(np.diff(t)>2)])
    false=0;start=indices
    for i in start:
        j=i+1
        while j<len(g) and p[j] and t[j]-t[j-1]<=2:j+=1
        # False episode has no positive future target anywhere during the run.
        false += int(not g.Label.iloc[i:j].eq(1).any())
    # Only observed evaluable time segments; missing-target gaps are not exposure.
    duration=float(np.diff(t)[np.diff(t)<=2].sum()) if len(t)>1 else 0.0
    return {'Alert_Episodes':len(start),'False_Alert_Episodes':false,'Exposure_Hours':duration/3600,
            'False_Alerts_Per_Hour':false/(duration/3600) if duration else np.nan,
            'FP_Timestamps_Per_Hour':int(((g.Label==0)&(g.Prediction==1)).sum())/(duration/3600) if duration else np.nan}
