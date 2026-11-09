"""SELF-IMPLEMENTED biased regularized matrix factorization with mini-batch SGD.

For an observed rating r_ui, prediction is mu + bu + bi + dot(pu, qi).
Loss = mean squared prediction error + L2 penalty on biases/factors.
Updates accumulate gradients for duplicate users/items within each batch.
No held-out ratings are used for fitting.
"""
from __future__ import annotations
import numpy as np
from recsys.models.base import Recommender
from recsys.data.dataset import TrainData

class BiasedMF(Recommender):
    name = 'BiasedMF (self-implemented)'
    stochastic = True
    def __init__(self, factors: int = 24, epochs: int = 12, lr: float = 0.02,
                 reg: float = 0.05, batch_size: int = 8192, seed: int = 42):
        super().__init__(factors=factors, epochs=epochs, lr=lr, reg=reg, batch_size=batch_size, seed=seed)
        self.factors, self.epochs, self.lr, self.reg = factors,epochs,lr,reg
        self.batch_size,self.seed = batch_size,seed
        if min(factors,epochs,batch_size)<=0 or lr<=0 or reg<0:
            raise ValueError('Invalid MF hyperparameters')
    def _fit(self, train: TrainData) -> None:
        rng=np.random.default_rng(self.seed)
        rows=train.ratings
        u=train.ids.user_index(rows['user_id']).astype(np.int64)
        i=train.ids.item_index(rows['item_id']).astype(np.int64)
        y=rows['rating'].to_numpy(dtype=np.float64)
        self.mu=float(y.mean()) if len(y) else 3.0
        self.p=rng.normal(0,0.08,(train.n_users,self.factors))
        self.q=rng.normal(0,0.08,(train.n_items,self.factors))
        self.bu=np.zeros(train.n_users)
        self.bi=np.zeros(train.n_items)
        self.loss_history=[]
        for _ in range(self.epochs):
            order=rng.permutation(len(y))
            for first in range(0,len(y),self.batch_size):
                idx=order[first:first+self.batch_size]
                uu,ii,yy=u[idx],i[idx],y[idx]
                pu,qi=self.p[uu].copy(),self.q[ii].copy()
                err=self.mu+self.bu[uu]+self.bi[ii]+np.sum(pu*qi,axis=1)-yy
                # Aggregate duplicates, then perform a per-entity mean SGD step.
                # Global batch averaging would make updates vanishingly small
                # for users/items that occur only a few times per batch.
                unique_u,inv_u,counts_u=np.unique(uu,return_inverse=True,return_counts=True)
                unique_i,inv_i,counts_i=np.unique(ii,return_inverse=True,return_counts=True)
                gu=np.zeros((len(unique_u),self.factors))
                gi=np.zeros((len(unique_i),self.factors))
                gbu=np.zeros(len(unique_u));gbi=np.zeros(len(unique_i))
                np.add.at(gu,inv_u,err[:,None]*qi+self.reg*pu)
                np.add.at(gi,inv_i,err[:,None]*pu+self.reg*qi)
                np.add.at(gbu,inv_u,err+self.reg*self.bu[uu])
                np.add.at(gbi,inv_i,err+self.reg*self.bi[ii])
                self.p[unique_u] -= self.lr*gu/counts_u[:,None]
                self.q[unique_i] -= self.lr*gi/counts_i[:,None]
                self.bu[unique_u] -= self.lr*gbu/counts_u
                self.bi[unique_i] -= self.lr*gbi/counts_i
            # Diagnostic training RMSE, not used to select hyperparameters.
            pred=self.mu+self.bu[u]+self.bi[i]+(self.p[u]*self.q[i]).sum(axis=1)
            self.loss_history.append(float(np.sqrt(np.mean((pred-y)**2))) if len(y) else 0)
    def score(self, users: np.ndarray) -> np.ndarray:
        return self.mu+self.bu[users,None]+self.bi[None,:]+self.p[users]@self.q.T
