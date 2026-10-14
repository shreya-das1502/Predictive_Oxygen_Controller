"""Research alert inference. Does not actuate oxygen delivery."""
import numpy as np
from .model import FittedModel
from .features import FeatureState
from .preprocessing import WaveformState
from .config import MAX_NUMERIC_GAP, SPO2_THRESHOLD

class Controller:
    def __init__(self,model):
        if model.threshold is None or not np.isfinite(model.threshold):raise ValueError('Model requires a saved validation-derived threshold')
        self.model=model;self.feature_state=FeatureState();self.waveform_state=WaveformState();self.last_time=None
    @classmethod
    def load(cls,path):return cls(FittedModel.load(path))
    def predict(self,measurement):
        t=float(measurement.get('time_s',np.nan))
        if not np.isfinite(t) or (self.last_time is not None and t<=self.last_time):raise ValueError('Pulse timestamps must be finite and strictly increasing; reset for a new recording')
        x=self.feature_state.update(measurement)
        self.last_time=t
        target=getattr(self.model,'target_definition',{})
        if target.get('name')=='new_onset':
            spo2=float(measurement.get('spo2',np.nan));st=float(measurement.get('spo2_time_s',np.nan))
            gap=target.get('maximum_numeric_gap_s',MAX_NUMERIC_GAP)
            threshold=target.get('threshold_spo2',SPO2_THRESHOLD)
            valid=np.isfinite(spo2) and 0<spo2<=100 and np.isfinite(st) and 0<=t-st<=gap
            if not valid or spo2<=threshold:
                return {'time_s':t,'score':None,'alert':False,'threshold':float(self.model.threshold),
                        'model':self.model.name,'version':self.model.version,'eligible':False,
                        'reason':'current_spo2_unknown' if not valid else 'current_spo2_not_above_threshold'}
        score=float(self.model.score(x[None,:])[0])
        return {'time_s':t,'score':score,'alert':bool(score>=self.model.threshold),'threshold':float(self.model.threshold),
                'model':self.model.name,'version':self.model.version,'eligible':True,'missing_features':int(np.isnan(x).sum())}
    def update_waveform(self,time_s,ppg,spo2=None,spo2_time_s=None):
        measurement=self.waveform_state.update(time_s,ppg)
        if measurement is None:return None
        if spo2 is not None:measurement.update(spo2=spo2,spo2_time_s=spo2_time_s)
        return self.predict(measurement)
