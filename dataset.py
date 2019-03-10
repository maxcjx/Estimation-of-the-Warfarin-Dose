import numpy as np
import csv

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def is_complete(row):
    if row[34] == 'NA' or row[4] == 'NA' or row[5] == 'NA' or row[6] == 'NA' or \
            is_number(row[34]) == False or is_number(row[5]) == False or is_number(row[6]) == False:
        return False
    return True

def age_bucket(age):
    end = 0
    for i, c in enumerate(age):
        if c.isdigit() == False:
            end = i
            break

    return int(age[0:end]) / 10


def dosage_bucket(weekly_dosage):
    daily_dosage = weekly_dosage / 7.0
    if daily_dosage < 3:
        return 0
    elif daily_dosage <= 7:
        return 1
    else:
        return 2


def dosage_reward(arm_chosen, label):
    if arm_chosen == dosage_bucket(label):
        return 0
    else:
        return -1


def load_data(file_name):
    f = open(file_name)
    reader = csv.reader(f)
    heading = next(reader)
    data = []
    labels = []
    for row in reader:
        if not is_complete(row):
            continue
        labels.append(float(row[34]))
        data.append(row)
    return data, np.array(labels)

'''
    Input: a row
    Output: a feature vector, ndarray

    row[1]: Gender
    row[2]: Race
    row[4]: Age
    row[5]: Height
    row[6]: Weight
    row[23]: Amiodarone
    row[24]: Carbamazepine
    row[25]: Phenytoin (Dilantin)
    row[26]: Rifampin or Rifampicin
    row[34]: Therapeutic Dose of Warfarin
'''
def feature_extractor(row):
    feature = []

    # Age in decades
    feature.append(age_bucket(row[4]))

    # Height in cm
    feature.append(float(row[5]))

    # Weight in kg
    feature.append(float(row[6]))

    # Race
    if row[2] == 'Asian':
        feature += [1, 0, 0]
    elif row[3] == 'Black or African American':
        feature += [0, 1, 0]
    else:
        feature += [0, 0, 1]

    # Enzyme inducer status
    feature.append(float(row[24] == '1' or row[25] == '1' or row[26] == '1'))

    # Amiodarone status
    feature.append(float(row[23] == '1'))

    # Bias
    feature.append(1)

    return np.array(feature)

def extract_features(data):
    return np.array([feature_extractor(row) for row in data])


def calc_regret(prediction, labels, reward_function):
    regret = 0
    for row in range(len(prediction)):
        regret += 0 - reward_function(prediction[row], labels[row])
    return regret


def calc_accuracy(prediction, labels, bucket_function):
    correct = sum([prediction[row] == bucket_function(labels[row]) for row in range(len(prediction))])
    return float(correct) / len(prediction)
