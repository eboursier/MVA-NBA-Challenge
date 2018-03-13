import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

STATS = np.array(['score', 'offensive rebound', 'defensive rebound', 'offensive foul',
 		 'defensive foul', 'assist', 'lost ball', 'steals', 'bad pass',
 		  'block', 'miss'])

# GLOBAL_STATS = 


def load_train(nrows=None, global_stats=False):
	train_label = pd.read_csv('dataset/challenge_output_data_training_file_nba_challenge.csv', sep=";", index_col=0, nrows=nrows)

	if global_stats:
		train = pd.read_csv('features/global_stats.csv', nrows=nrows)
	else:
		train = pd.read_csv('dataset/train.csv', nrows=nrows)

	return train, train_label

def load_test(nrows=None, global_stats=False):
	if global_stats:
		test = pd.read_csv('features/global_stats_test.csv', nrows=nrows)
	else:
		test = pd.read_csv('dataset/test.csv', nrows=nrows)

	return test

def subsampling(x, t=60):
    """
    We aggregate the data over x each t seconds. This way, we reduce the dimension of the data.
    x is set as the array train_array just above
    """
    samples = t*np.arange(1, 1440//t+1)-1
    return x[:, samples, :]

def plot_stats(game_idx, med=True, subsampling=1):

    if med:
        med_game = np.median(train_array, axis=0)
    game = train_array[game_idx]
    for i in range(11):
        plt.figure(i)
        if med:
            plt.plot(med_game[:, i], label='median game')
        plt.plot(game[:, i], label='chosen game')
        plt.title(stats[i])
        plt.legend()
    plt.show()

def delta(X):
    """
    for an input of dimension (n_samples, timesteps, input_dim), it returns the difference corresponding time series
    of size (n_samples, timesteps, input_dim).
    """
    X_translated = np.zeros(X.shape)
    X_translated[:, 1:, :] = X[:, :-1, :]
    return X-X_translated


def compute_global_stats(X, keep_stats):
	"""
	We add global interesting stats as the total number of offensive rebounds or the field goals % of each team as a time series.
	Keep stats represent as strings the stats we kept from the initial series.
	"""

	# we will use the delta array
	diff = delta(X)
	stats = []
	n = X.shape[0]
	del X

	stats_guest = np.maximum(diff, 0)
	stats_home = np.maximum(-diff, 0)

	del diff

	specific_stats = ['score', 'miss'] 
	# first compute the number of points of each team
	if 'score' in keep_stats:
		print('Computing points')
		score_idx = np.where(keep_stats=='score')[0][0]
		points_guest = np.cumsum(stats_guest[:, :, score_idx], axis=1)
		Z = points_guest.reshape(n, -1, 1)
		stats.append('points_guest')
		del points_guest

		points_home = np.cumsum(stats_home[:, :, score_idx], axis=1)
		Z = np.concatenate((Z, points_home.reshape(n, -1, 1)), axis=2)
		stats.append('points_home')
		del points_home
	    
	# compute the total number of misses to compute other stats
	if 'miss' in keep_stats:
		print('computing misses')
		miss_idx = np.where(keep_stats=='miss')[0][0] 
		misses_guest = np.cumsum(stats_guest[:, :, miss_idx], axis=1)
		misses_home = np.cumsum(stats_home[:, :, miss_idx], axis=1)
	    
		# compute FG %
		# the case of two made FTs is counted as a made shot, because we do not have
		# the information to avoid this approximation
		if 'score' in keep_stats:
			print('computing FGs %')
			madeshots_guest = np.cumsum(stats_guest[:, :, score_idx]>=2, axis=1)
			FG_guest = madeshots_guest/(madeshots_guest+misses_guest+1e-8)
			Z = np.concatenate((Z, FG_guest.reshape(n, -1, 1)), axis=2)
			stats.append('FG_guest')
			del madeshots_guest, FG_guest            

			madeshots_home = np.cumsum(stats_home[:, :, score_idx]>=2, axis=1)
			FG_home = madeshots_home/(madeshots_home+misses_home+1e-8)
			Z = np.concatenate((Z, FG_home.reshape(n, -1, 1)), axis=2)
			stats.append('FG_home')
			del madeshots_home, FG_home
	        
		# compute the Offensive rebound ratio
		if 'offensive rebound' in keep_stats:
			print('Computing offensive rebounds efficiency')
			off_reb_idx = np.where(keep_stats=='offensive rebound')[0][0] 
			off_reb_guest = np.cumsum(stats_guest[:, :, off_reb_idx], axis=1)
			off_effi_guest = off_reb_guest/(misses_guest+1e-8)
			Z = np.concatenate((Z, off_effi_guest.reshape(n, -1, 1)), axis=2)
			stats.append('off_reb_effi_guest')
			del off_reb_guest, off_effi_guest, misses_guest

			off_reb_home = np.cumsum(stats_home[:, :, off_reb_idx], axis=1)
			off_effi_home = off_reb_home/(misses_home+1e-8)
			Z = np.concatenate((Z, off_effi_home.reshape(n, -1, 1)), axis=2)
			stats.append('off_reb_effi_home')
			del off_reb_home, off_effi_home, misses_home

	# compute total number of all stats different from score and misses
	for s in np.setdiff1d(keep_stats, specific_stats):
		print('computing global', s)
		s_idx = np.where(keep_stats==s)[0][0]
		s_guest = np.cumsum(stats_guest[:, :, s_idx], axis=1)            
		Z = np.concatenate((Z, s_guest.reshape(n, -1, 1)), axis=2)
		stats.append((s + '_guest'))
		del s_guest

		s_home = np.cumsum(stats_home[:, :, s_idx], axis=1)
		Z = np.concatenate((Z, s_home.reshape(n, -1, 1)), axis=2)
		stats.append((s + '_home'))
		del s_home

	return Z, np.array(stats)


def preprocessing(train_array, train_label, sampling_rate=60, keep_stats='all'):
	"""
	return X_train, y_train
	"""
	# remove defensive foul stats
	if keep_stats != 'all':
		keep_stats = np.array(keep_stats)

	ID = train_array.ID
	train_array = train_array.values[:, 1:] # we remove the ID information
	train_array = train_array.reshape(train_array.shape[0], 1440, train_array.shape[1]//1440)

	if keep_stats != 'all':
		train_array = train_array[:, :, keep_stats]
		
	
	X = subsampling(train_array, sampling_rate).reshape(train_array.shape[0], -1)
    del train_array
	X = pd.DataFrame(X, index=ID)
	X = X.join(train_label)
	y = X.pop('label')

	return X, y


if __name__ == '__main__':

	print('Load data ...')
	train, train_label = load_data(nrows=None)
	ID = train.ID
	print('data successfully loaded')
	keep_stats=np.arange(0, 11)
	keep_stats = keep_stats[keep_stats!=4]
	train = train.values[:, 1:] # we remove the ID information
	train = train.reshape(train.shape[0], 1440, train.shape[1]//1440)

	train = train[:, :, keep_stats]

	keep_stats = STATS[keep_stats]
	print('start computing global stats')
	train, stats = compute_global_stats(train, keep_stats)
	train = train.reshape(train.shape[0], -1)
	train = pd.DataFrame(train, index=ID)
	print('Saving features')
	train.to_csv('./features/global_stats.csv')
	print('successfully saved')
