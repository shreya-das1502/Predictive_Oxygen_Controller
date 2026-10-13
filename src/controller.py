"""Research alert inference. Does not actuate oxygen delivery."""
import numpy as np
from .model import FittedModel
from .features import FeatureState
from .preprocessing import WaveformState

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
        score=float(self.model.score(x[None,:])[0]);self.last_time=t
        return {'time_s':t,'score':score,'alert':bool(score>=self.model.threshold),'threshold':float(self.model.threshold),
                'model':self.model.name,'version':self.model.version,'missing_features':int(np.isnan(x).sum())}
    def update_waveform(self,time_s,ppg):
        measurement=self.waveform_state.update(time_s,ppg)
        return None if measurement is None else self.predict(measurement)
