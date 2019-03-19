import numpy as np
from dataset import *
import copy
from collections import defaultdict
from sklearn import linear_model

def normalize(features):
    features = copy.deepcopy(features)
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    return (features - mean) / (std + 1e-10)

def computeReward(x, featureList, labelList, lamda):
    featureArray = np.array(featureList) 
    labelArray = np.array(labelList)
    clf = linear_model.Lasso(alpha = lamda, fit_intercept=False, max_iter= 1000000)
    clf.fit(featureArray, labelArray)

    #print(clf.coef_)
    #print(np.dot(clf.coef_, x.T))
    return np.dot(clf.coef_, x.T)

'''
    LASSO for the bandit setting
'''
def LASSOBandit(features, labels, num_arms, reward_function, q, h, lamda_1, lamda_2):
    #features = normalize(features)
    num_rows, num_features = features.shape
    start_lamda2 = lamda_2

    '''
        Constrcut forced-sample set
    '''
    forcedArm = np.zeros(num_rows)
    forced_featureDict = defaultdict(list)
    forced_labelDict = defaultdict(list)

    power = 0
    for row in range(num_rows):
        if row >= (2**power-1)*3*q-1 and row <= (2**power-1)*3*q+3*q-1:
            arm = (row-(2**power-1)*3*q)/q+1
            forcedArm[row] = arm
            if row == (2**power-1)*3*q+3*q-1:
                power +=1

    all_featureDict = defaultdict(list)
    all_labelDict = defaultdict(list)
    prediction = []

    for row in range(num_rows):
        arm_chosen = forcedArm[row]-1
        if forcedArm[row] == 0:
            forced_maxReward = float("-inf")
            estimated_rewards = []
            for arm in range(num_arms):
                estimated_reward = computeReward(features[row], forced_featureDict[arm], forced_labelDict[arm], lamda_1)
                estimated_rewards.append(estimated_reward)
                forced_maxReward = max(forced_maxReward, estimated_reward)

            #print(forced_maxReward)
            armSet = []
            for arm in range(num_arms):
                if estimated_rewards[arm] >= forced_maxReward-h*1.0/2:
                    armSet.append(arm)

            all_maxReward = float("-inf")
            for selected_arm in armSet:
                estimated_reward = computeReward(features[row], all_featureDict[selected_arm], all_labelDict[selected_arm], lamda_2)
                if all_maxReward < estimated_reward:
                    all_maxReward = estimated_reward
                    arm_chosen = selected_arm

        all_featureDict[arm_chosen].append(features[row])
        reward = reward_function(arm_chosen, labels[row])
        all_labelDict[arm_chosen].append(reward)

        if forcedArm[row] != 0:
	        forced_featureDict[arm_chosen].append(features[row])
	        forced_labelDict[arm_chosen].append(reward)

        lamda_2 = start_lamda2*np.sqrt((np.log2(row+1)+np.log2(num_features))*1.0/(row+1))
        prediction.append(arm_chosen)
    return prediction
