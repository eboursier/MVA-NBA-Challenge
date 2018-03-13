import matplotlib.pyplot as plt
import pandas as pd

def plot_game(X, Y, idx, events = ['score', 'assist', 'miss', 'lost ball']):

    label = Y.loc[idx]['label']
    winner = 'visitor' if label == 1 else 'home'

    fig = plt.figure(figsize = (13, 6))

    for feat in events:
        X.loc[idx][feat].plot(label = feat)

    plt.xlabel('seconds')

    plt.title('Game {}, won by {}'.format(idx, winner))

    plt.legend()
    plt.grid()
    plt.show()
