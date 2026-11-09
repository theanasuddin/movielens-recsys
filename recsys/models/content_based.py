"""Positive-history user profiles over multi-hot genre + optional release decade.
No protected user attributes are used. Item metadata are public catalogue features.
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize
from recsys.data.movielens import GENRES
from recsys.models.base import Recommender
from recsys.data.dataset import TrainData

class ContentProfile(Recommender):
    name='ContentProfile'
    def __init__(self, movies, min_rating: int=4, year_weight: float=0.):
        super().__init__(min_rating=min_rating,year_weight=year_weight)
        self.movies=movies
        self.min_rating=min_rating
        self.year_weight=year_weight
        if year_weight<0:raise ValueError('year_weight must be nonnegative')
    def _fit(self, train: TrainData) -> None:
        meta=self.movies.set_index('item_id').reindex(train.ids.items)
        genre=np.column_stack([meta['genres'].fillna('').str.split('|').map(lambda g:float(s in g)).to_numpy() for s in GENRES]).astype(np.float32)
        if self.year_weight:
            years=meta['year'].astype(float).fillna(1990).to_numpy()
            decade=np.clip(((years-1900)//10).astype(int),0,12)
            decade_features=np.eye(13,dtype=np.float32)[decade]*self.year_weight
            genre=np.column_stack([genre,decade_features])
        self.features=normalize(genre,axis=1)
        x=train.matrix('binary',self.min_rating)
        profiles=x @ self.features
        self.profiles=normalize(profiles,axis=1)
    def score(self,users:np.ndarray)->np.ndarray:
        return np.asarray(self.profiles[users]@self.features.T)
