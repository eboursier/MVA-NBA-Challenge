import numpy as np
import pandas as pd
import sys

import matplotlib.pyplot as plt

from data_transform import *
from plots import *

def compute_instant_stats(df,
                          original_feats = ['assist', 'bad pass', 'block', 'defensive rebound',
                                           'lost ball', 'miss', 'offensive foul', 'offensive rebound',
                                           'score', 'steals']):

    """Computes instant values for each statistics from
    the original values of differences"""

    T = 1440
    N = df.shape[0] // T

    for feature in original_feats:
        X_feat = df[feature].as_matrix().reshape((N, T))

        X_previous = np.zeros_like(X_feat)
        X_previous[:,1:] = X_feat[:,:-1]

        X_diffs = X_feat - X_previous

        df[feature + '_inst'] = X_diffs.reshape(-1)

    return df

def denoise_scores(score_matrix, max_variation = 5):
    """Takes the cumulative (N,T) matrix of scores and removes any
    variation that is too high and is taken into account as noise."""

    N, T = score_matrix.shape

    for i in range(N):
        if i % 1000 == 0:
            print ('Processing row {}/{}'.format(i, N))

        x_prev = 0

        for t in range(1, T):
            if np.abs(score_matrix[i,t] - x_prev) > max_variation:
                score_matrix[i,t] = x_prev

            x_prev = score_matrix[i,t]

        if (score_matrix[i] == 0).all():
            print ('Game #{} is zeroed out'.format(i))

    return score_matrix


if __name__ == '__main__':

	mode = sys.argv[1]

	
	if mode=='train':
		print('Load train data ...')
		X_1 = pd.read_csv('./data/train.csv', index_col=0)
		Y = pd.read_csv('./data/challenge_output_data_training_file_nba_challenge.csv', sep=";", index_col=0)
	elif mode=='test':
		print('Load test data ...')
		X_1 = pd.read_csv('./data/test.csv', index_col=0)
	else:
		print('You must enter as argument train or test')

	print('building multi-index')
	# building multi-index
	cols = X_1.columns.tolist()
	multi_index = [(c.split('_')[0], int(c.split('_')[1])) for c in cols]
	index = pd.MultiIndex.from_tuples(multi_index)
	X = X_1.copy()
	X.columns = index
	X_stacked = X.stack()
	X_stacked = X_stacked.drop('defensive foul', 1) # always equal to 0
	N = X_1.shape[0]
	T = 1440
	original_feats = X_stacked.columns.tolist()

	# avoiding to consume too much memory
	del X_1
	del X

	print('compute the instant features from each event')
	X_inst = compute_instant_stats(X_stacked, original_feats)

	print('remove games with too little data')
	score_count = X_inst[X_inst['score_inst'] != 0].groupby(level = 0).count()['score']
	del X_inst

	X_unstacked = X_stacked.unstack()
	del X_stacked

	X_unstacked['score_count'] = score_count

	if mode=='train':
		X_unstacked = X_unstacked[X_unstacked['score_count'] > 30]

	del X_unstacked['score_count']
	
	if mode=='train':
		Y_new = Y.loc[X_unstacked.index.tolist()]
		del Y

	print('denoising score')
	score_matrix = X_unstacked['score'].as_matrix()
	scores_denoised = denoise_scores(score_matrix, 5)
	X_unstacked['score'] = scores_denoised

	print('Saving clean data')
	if mode=='train':
		X_unstacked.to_csv('./data/train_clean.csv')
		Y_new.to_csv('./data/train_label_clean.csv')
	elif mode=='test':
		X_unstacked.to_csv('./data/test_clean.csv')
	print('successfully saved')
