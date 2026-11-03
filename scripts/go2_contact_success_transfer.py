"""157 pure normalized70 -> residual, fixed cosine features and one linear fit."""
import numpy as np


def feature(x,frequency,phase):
    return np.sqrt(2/frequency.shape[1])*np.cos(np.asarray(x,np.float64)@frequency+phase)


class ContactSuccessActor:
    def __init__(self,path):
        with np.load(path) as z:self.frequency=z['frequency'].copy();self.phase=z['phase'].copy();self.coefficient=z['coefficient'].copy()
    def predict(self,observation,deterministic=True):
        assert deterministic
        x=np.asarray(observation,np.float32);assert x.shape==(70,)
        return np.clip(feature(x,self.frequency,self.phase)@self.coefficient,-1,1),None
