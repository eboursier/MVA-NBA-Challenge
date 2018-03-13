import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sys

STATS = np.array(['score', 'offensive rebound', 'defensive rebound', 'offensive foul',
 		 'defensive foul', 'assist', 'lost ball', 'steals', 'bad pass',
 		  'block', 'miss'])

# GLOBAL_STATS = 


def subsampling(x, t=60):
    """
    We aggregate the data over x each t seconds. This way, we reduce the dimension of the data.
    x is set as the array train_array just above
    """
    samples = t*np.arange(1, 1440//t+1)-1
    return x[:, samples, :]

def compute_average_possessions(p):
	n = p.shape[0]
	average_poss_guest = np.zeros_like(p)
	average_poss_home = np.zeros_like(p)

	for i in range(n):
		if i % 1000 == 0:
			print ('Processing row {}/{}'.format(i, n))


		n_h = 0
		n_g = 0
		t_h = 0
		t_g = 0
		b = 0
		time = 0

		for t in range(1440):
			time += 1
			if p[i, t] > 0:
				# guest got the ball
				if b==1:
					# case guest already had the possession
					t_g += time
					n_g += 1
				if b==-1:
					# case home had the previous possession
					t_h += time
					n_h += 1
				# reset the clock of the possession and give the ball to guest
				b = 1
				time = 0
			elif p[i, t] < 0:
				# home got the ball
				if b==1:
					# case guest had the previous possession
					t_g += time
					n_g += 1
				if b==-1:
					# case home already had the previous possession
					t_h += time
					n_h += 1
				# reset the clock of the possession and give the ball to home
				b = -1
				time = 0

			average_poss_guest[i, t] = t_g/(n_g+1e-8)
			average_poss_home[i, t] = t_h/(n_h+1e-8)

	return average_poss_guest, average_poss_home


def compute_stats(X, statistics):
	"""
	We add global interesting stats as the total number of offensive rebounds or the field goals % of each team as a time series.
	Keep stats represent as strings the stats we kept from the initial series.
	"""


	# we will use the delta array
	n = X.shape[0]
	stats = statistics.copy()

	# compute the total number of misses to compute other stats

	print('computing misses')
	misses_guest = np.cumsum(np.maximum(X[:, :, stats.index('miss_inst')], 0), axis=1)
	misses_home = np.cumsum(np.maximum(-X[:, :, stats.index('miss_inst')], 0), axis=1)
	    
	# compute FG %
	# the case of two made FTs is counted as a made shot, because we do not have
	# the information to avoid this approximation
	print('computing FGs %')
	madeshots_guest = np.cumsum(X[:, :, stats.index('score_inst')]>=2, axis=1)
	FG_guest = madeshots_guest/(madeshots_guest+misses_guest+1e-8)
	X = np.concatenate((X, FG_guest.reshape(n, -1, 1)), axis=2)
	stats.append('FG_guest')

	del madeshots_guest, FG_guest            

	madeshots_home = np.cumsum(-X[:, :, stats.index('score_inst')]>=2, axis=1)
	FG_home = madeshots_home/(madeshots_home+misses_home+1e-8)
	X = np.concatenate((X, FG_home.reshape(n, -1, 1)), axis=2)
	stats.append('FG_home')
	del madeshots_home, FG_home
	        
	# compute the Offensive rebound ratio
	print('Computing offensive rebounds efficiency')
	off_reb_guest = np.cumsum(np.maximum(X[:, :, stats.index('offensive rebound_inst')], 0), axis=1)
	off_effi_guest = off_reb_guest/(misses_guest+1e-8)
	X = np.concatenate((X, off_effi_guest.reshape(n, -1, 1)), axis=2)
	stats.append('off_reb_effi_guest')
	del off_reb_guest, off_effi_guest, misses_guest

	off_reb_home = np.cumsum(np.maximum(-X[:, :, stats.index('offensive rebound_inst')], 0), axis=1)
	off_effi_home = off_reb_home/(misses_home+1e-8)
	X = np.concatenate((X, off_effi_home.reshape(n, -1, 1)), axis=2)
	stats.append('off_reb_effi_home')
	del off_reb_home, off_effi_home, misses_home


	print('computing average time possession')
	# compute the mean time for a possession of each team
	# to do so, we first compute a new vector which indicates any change of possession
	# team A has a new possession for each rebound/steal of team A or each score change (without offensive rebound of the other team in case of missed free throw)/lost ball/offensive foul
	# of team B

	# the problem is that we can't take the defensive fouls as new possessions since the data is not given for defensive fouls
	# because of that, we can observe possessions of more than 24 s

	secs = np.arange(1, 1441)

	possession_guest = (X[:, :, stats.index('offensive rebound_inst')]>0) | (X[:, :, stats.index('defensive rebound_inst')]>0) | (X[:, :, stats.index('steals_inst')]>0)
	possession_guest = possession_guest | (X[:, :, stats.index('lost ball_inst')]>0) | (X[:, :, stats.index('offensive foul_inst')]<0)
	# case of score change without any offensive rebound of the opposite team
	possession_guest = possession_guest | ((X[:, :, stats.index('score_inst')]<0) & (X[:, :, stats.index('offensive rebound_inst')]>=0))
	possession_guest = possession_guest.astype(int)

	possession_home = (-X[:, :, stats.index('offensive rebound_inst')]>0) | (-X[:, :, stats.index('defensive rebound_inst')]>0) | (-X[:, :, stats.index('steals_inst')]>0)
	possession_home = possession_home | (-X[:, :, stats.index('lost ball_inst')]>0) | (-X[:, :, stats.index('offensive foul_inst')]<0)
	# case of score change without any offensive rebound of the opposite team
	possession_home = possession_home | ((-X[:, :, stats.index('score_inst')]<0) & (-X[:, :, stats.index('offensive rebound_inst')]>=0))
	possession_home = possession_home.astype(int)

	possessions = possession_guest - possession_home
	del possession_home, possession_guest

	secs = np.arange(1, 1441)
	possessions = possessions*secs
	average_poss_guest, average_poss_home = compute_average_possessions(possessions)
	del possessions

	X = np.concatenate((X, average_poss_guest.reshape(n, -1, 1)), axis=2)
	stats.append('average_poss_guest')
	del average_poss_guest

	X = np.concatenate((X, average_poss_home.reshape(n, -1, 1)), axis=2)
	stats.append('average_poss_home')
	del average_poss_home

	inst_idx = np.where(['inst' in s for s in stats])
	X = np.delete(X, inst_idx, 2)

	stats = list(np.delete(np.array(stats), inst_idx))

	return X, stats


if __name__ == '__main__':

	mode = sys.argv[1]

	
	if mode=='train':
		print('Load train data ...')
		X_0 = pd.read_csv('./data/train_clean.csv', index_col=0, skiprows=[1, 2])
	elif mode=='test':
		print('Load test data ...')
		X_0 = pd.read_csv('./data/test_clean.csv', index_col=0, skiprows=[1, 2])
	else:
		print('You must enter as argument train or test')

	print('rename columns')
	cols = X_0.columns.tolist()
	multi_index = [c.split('.')[0] for c in cols]
	multi_index = [(multi_index[i], i%1440) for i in range(len(multi_index))]
	stats = [multi_index[1440*i][0] for i in range(len(multi_index)//1440)]
	del multi_index

	print('Transform dataframe into usable matrix')
	idx = X_0.index
	Z = X_0.values.reshape(X_0.shape[0], -1, 1440)
	del X_0
	Z = np.swapaxes(Z, 1, 2)

	Z, stats = compute_stats(Z, stats)
	print(Z.shape)
	print(stats)

	print('last transformations')
	Z = np.swapaxes(Z,1,2)
	Z = Z.reshape(Z.shape[0], -1)
	Z = pd.DataFrame(Z, index=idx)

	index = pd.MultiIndex.from_product([stats, np.arange(0, 1440)])
	Z.columns = index
	print('saving the data')
	if mode=='train':
		Z.to_csv('./data/train_engineered.csv')
	elif mode=='test':
		Z.to_csv('./data/test_engineered.csv')
	print('successfully saved')