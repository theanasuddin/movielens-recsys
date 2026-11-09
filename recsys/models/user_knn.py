"""User-neighbour collaborative filtering; ratings >= min_rating are positives.

Similarity is computed from TRAIN ONLY. k highest nonnegative neighbours
vote for each candidate, optionally normalized by sum of similarity weights.
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
from recsys.models.base import Recommender
from recsys.data.dataset import TrainData

class UserKNN(Recommender):
    name = 'UserKNN'
    def __init__(self, k: int = 40, similarity: str = 'cosine', min_rating: int = 4):
        super().__init__(k=k, similarity=similarity, min_rating=min_rating)
        if similarity not in ('cosine', 'jaccard') or k < 1:
            raise ValueError('similarity must be cosine/jaccard, k >= 1')
        self.k, self.similarity, self.min_rating = k, similarity, min_rating
    def _fit(self, train: TrainData) -> None:
        self.x = train.matrix('binary', self.min_rating).tocsr()
        co = (self.x @ self.x.T).toarray().astype(np.float32)
        sizes = np.diag(co).copy()
        if self.similarity == 'jaccard':
            den = sizes[:,None]+sizes[None,:]-co
        else:
            den = np.sqrt(sizes[:,None]*sizes[None,:])
        sim = np.divide(co,den,out=np.zeros_like(co),where=den>0)
        np.fill_diagonal(sim,0)
        k = min(self.k, max(train.n_users-1,1))
        self.neighbors = sp.lil_matrix((train.n_users,train.n_users),dtype=np.float32)
        for u in range(train.n_users):
            ids = np.argpartition(-sim[u], k-1)[:k]
            ids = ids[sim[u,ids]>0]
            if len(ids): self.neighbors[u,ids] = sim[u,ids]
        self.neighbors = self.neighbors.tocsr()
    def score(self, users: np.ndarray) -> np.ndarray:
        weights = self.neighbors[users]
        votes = (weights @ self.x).toarray()
        denom = np.asarray(weights.sum(axis=1)).reshape(-1,1)
        return np.divide(votes,denom,out=np.zeros_like(votes),where=denom>0)
