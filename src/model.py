"""Fixed models with training-only transformations and reproducible persistence."""
import json
from pathlib import Path
import joblib
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from .config import MODEL_CONFIGS, FEATURES, VERSION

class ELM:
    """Weighted ridge output on fixed random tanh hidden features; uncalibrated score."""
    def __init__(self,hidden_nodes=50,ridge=1.0,positive_weight=5.0,seed=42):
        self.hidden_nodes=hidden_nodes;self.ridge=ridge;self.positive_weight=positive_weight;self.seed=seed
    def fit(self,X,y):
        X=np.asarray(X,float);y=np.asarray(y,float)
        if X.ndim!=2 or len(X)!=len(y) or not len(y) or not np.isfinite(X).all() or not np.isin(y,[0,1]).all():raise ValueError('ELM needs finite X and binary aligned labels')
        if self.ridge<=0 or self.hidden_nodes<1 or self.positive_weight<=0:raise ValueError('ELM regularization, units and weight must be positive')
        rng=np.random.default_rng(self.seed)
        self.weights=rng.normal(size=(X.shape[1],self.hidden_nodes));self.bias=rng.uniform(-1,1,self.hidden_nodes)
        H=np.tanh(X@self.weights+self.bias);sw=np.sqrt(np.where(y==1,self.positive_weight,1))
        A=H*sw[:,None];b=y*sw
        self.beta=cho_solve(cho_factor(A.T@A+self.ridge*np.eye(self.hidden_nodes),lower=True),A.T@b)
        return self
    def score(self,X):return np.tanh(np.asarray(X)@self.weights+self.bias)@self.beta

class FittedModel:
    def __init__(self,name):
        if name not in MODEL_CONFIGS:raise ValueError(f'Unknown model: {name}')
        self.name=name;self.features=list(FEATURES);self.config=dict(MODEL_CONFIGS[name]);self.version=VERSION
        self.imputer=SimpleImputer(strategy='median',keep_empty_features=True)
        self.scaler=StandardScaler()
        self.estimator={'ELM':ELM,'Logistic':LogisticRegression,'GaussianNB':GaussianNB,'Tree':DecisionTreeClassifier}[name](**self.config)
    def fit(self,X,y):
        X=np.asarray(X,float);y=np.asarray(y,int)
        if X.ndim!=2 or X.shape[1]!=len(self.features) or np.isinf(X).any():raise ValueError('Invalid training feature matrix')
        if len(np.unique(y))!=2:raise ValueError('Training recordings must contain both classes')
        self.estimator.fit(self.scaler.fit_transform(self.imputer.fit_transform(X)),y)
        self.threshold=None
        return self
    def score(self,X):
        X=np.asarray(X,float)
        if X.ndim!=2 or X.shape[1]!=len(self.features) or np.isinf(X).any():raise ValueError('Invalid inference feature matrix')
        Z=self.scaler.transform(self.imputer.transform(X))
        scores=self.estimator.score(Z) if self.name=='ELM' else self.estimator.predict_proba(Z)[:,1]
        if not np.isfinite(scores).all():raise ValueError('Non-finite model output')
        return scores
    def save(self,path,metadata):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        self.target_definition=metadata.get('target',{'name':'future_low'})
        joblib.dump(self,path)
        m={**metadata,'version':self.version,'model':self.name,'features':self.features,'hyperparameters':self.config,'threshold':self.threshold,
           'score_semantics':'uncalibrated ridge score' if self.name=='ELM' else 'uncalibrated classifier probability (class weighting/prior shift)'}
        path.with_suffix('.json').write_text(json.dumps(m,indent=2))
    @staticmethod
    def load(path):
        # Only load artifacts produced locally by this trusted pipeline.
        obj=joblib.load(path)
        if not isinstance(obj,FittedModel) or obj.version!=VERSION or obj.features!=FEATURES:raise ValueError('Incompatible model artifact')
        return obj
