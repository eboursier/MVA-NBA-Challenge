import numpy as np
import pandas as pd

def compute_instant_stats(df,
                          original_feats = ['assist', 'bad pass', 'block', 'defensive rebound',
                                           'lost ball', 'miss', 'offensive foul', 'offensive rebound',
                                           'score', 'steals']):

    """Computes instant values for each statistics from
    the original values of differences"""

    T = df.loc[0].shape[0]
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
