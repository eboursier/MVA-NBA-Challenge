import numpy as np
            
def score_function(y_true, y_pred):
    score = 0
    length1 = y_true.shape[0]
    for i in range(length1):
        if y_pred[i, :] == y_true[i, :]:
            score += 1
    return float(score)/float(length1)

