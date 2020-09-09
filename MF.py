from utils import *
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import train

r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']

ratings_base = pd.read_csv(rating_train, sep='\t', names=r_cols, encoding='latin_1')
ratings_test = pd.read_csv(rating_test, sep='\t', names=r_cols, encoding='latin_1')

rate_train = ratings_base.values
rate_test = ratings_test.values

# indices start from 0
rate_train[:, :2] -= 1
rate_test[:, :2] -= 1

rs = train.MF(rate_train, K = 10, lam=0.1, print_every=10, learning_rate=0.75, max_iter=100, user_based=1)
rs.fit()