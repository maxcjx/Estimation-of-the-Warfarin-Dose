import numpy as np
from dataset import *
import copy

def normalize(features):
    features = copy.deepcopy(features)
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    return (features - mean) / (std + 1e-10)

'''
    LinUCB with disjoint linear models.
'''
def LinUCB(features, labels, num_arms, reward_function, alpha):
    # features = normalize(features)
    num_rows, num_features = features.shape
    A = [np.identity(num_features) for _ in range(num_arms)]
    b = [np.zeros(num_features) for _ in range(num_arms)]
    prediction = []
    for row in range(num_rows):
        p = [None for _ in range(num_arms)]
        for arm in range(num_arms):
            A_inv = np.linalg.inv(A[arm])
            theta = np.dot(A_inv, b[arm])
            p[arm] = np.dot(theta, features[row]) + \
                     np.sqrt(np.dot(np.dot(features[row].T, A_inv), features[row]))
        arm_chosen = int(np.argmax(p))
        reward = reward_function(arm_chosen, labels[row])
        A[arm_chosen] += np.outer(features[row].T, features[row])
        b[arm_chosen] += alpha * reward * features[row]
        prediction.append(arm_chosen)
    return prediction