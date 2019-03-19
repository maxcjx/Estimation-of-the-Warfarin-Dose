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

def all_feature_extractor(row):
    feature = []

    # gender 
    if row[1] == 'male':
        feature.append(0)
    else:
        feature.append(1)

    # Race
    if row[2] == 'Asian':
        feature += [1, 0, 0]
    elif row[3] == 'Black or African American':
        feature += [0, 1, 0]
    else:
        feature += [0, 0, 1]

    # Age in decades
    feature.append(age_bucket(row[4]))

    # Height in cm
    feature.append(float(row[5]))

    # Weight in kg
    feature.append(float(row[6]))

    # Indication to treatment
    #feature.append(float(row[7]))

    # If cancer
    if row[8] == 'Cancer':
        feature += [1, 0, 0]
    elif row[8] == 'No Cancer':
        feature += [0, 1, 0]
    else:
        feature += [0, 0, 1]

    # Diabetes ~ Valve Replacement
    for i in range(9, 12):
        if row[i] == '1':
            feature += [1, 0, 0]
        elif row[i] == '0':
            feature += [0, 1, 0]
        else:
            feature += [0, 0, 1]

    # Aspirin ~ Herbal Medications, Vitamins, Supplements
    for i in range(13, 31):
        if row[i] == '1':
            feature += [1, 0, 0]
        elif row[i] == '0':
            feature += [0, 1, 0]
        else:
            feature += [0, 0, 1]      

    # smoker
    if row[36] == '1':
        feature += [1, 0, 0]
    elif row[36] == '0':
        feature += [0, 1, 0]
    else:
        feature += [0, 0, 1]

    # Cyp2C9 genotypes
    if row[37] == '*1/*1':
        feature += [1, 0, 0, 0, 0, 0, 0]
    elif row[37] == '*1/*2':
        feature += [0, 1, 0, 0, 0, 0, 0]   
    elif row[37] == '*1/*3':
        feature += [0, 0, 1, 0, 0, 0, 0]   
    elif row[37] == '*2/*2':
        feature += [0, 0, 0, 1, 0, 0, 0]   
    elif row[37] == '*2/*3':
        feature += [0, 0, 0, 0, 1, 0, 0]   
    elif row[37] == '*3/*3':
        feature += [0, 0, 0, 0, 0, 1, 0]   
    else :
        feature += [0, 0, 0, 0, 0, 0, 1]

    # VKORC1 genotype: -1639 G>A (3673)
    if row[41] == 'A/A':
        feature += [1, 0, 0, 0]
    elif row[41] == 'A/G':
        feature += [0, 1, 0, 0]
    elif row[41] == 'G/G':
        feature += [0, 0, 1, 0]
    else:
        feature += [0, 0, 0, 1]

    # VKORC1 genotype: 497T>G (5808)
    if row[43] == 'G/G':
        feature += [1, 0, 0, 0]
    elif row[43] == 'G/T':
        feature += [0, 1, 0, 0]
    elif row[43] == 'T/T':
        feature += [0, 0, 1, 0]
    else:
        feature += [0, 0, 0, 1]

    # VKORC1 genotype: 1173 C>T(6484)
    if row[45] == 'C/C':
        feature += [1, 0, 0, 0]
    elif row[45] == 'C/T':
        feature += [0, 1, 0, 0]
    elif row[45] == 'T/T':
        feature += [0, 0, 1, 0]
    else:
        feature += [0, 0, 0, 1]

    # VKORC1 genotype: 1542G>C (6853)
    if row[47] == 'C/C':
        feature += [1, 0, 0, 0]
    elif row[47] == 'C/G':
        feature += [0, 1, 0, 0]
    elif row[47] == 'G/G':
        feature += [0, 0, 1, 0]
    else:
        feature += [0, 0, 0, 1]

    # VKORC1 genotype: 3730 G>A (9041)
    if row[49] == 'A/A':
        feature += [1, 0, 0, 0]
    elif row[49] == 'A/G':
        feature += [0, 1, 0, 0]
    elif row[49] == 'G/G':
        feature += [0, 0, 1, 0]
    else:
        feature += [0, 0, 0, 1]

    # VKORC1 genotype: 2255C>T (7566)
    if row[51] == 'C/C':
        feature += [1, 0, 0, 0]
    elif row[51] == 'C/T':
        feature += [0, 1, 0, 0]
    elif row[51] == 'T/T':
        feature += [0, 0, 1, 0]
    else:
        feature += [0, 0, 0, 1]

    # VKORC1 genotype: -4451 C>A (861)
    if row[53] == 'A/A':
        feature += [1, 0, 0, 0]
    elif row[53] == 'A/C':
        feature += [0, 1, 0, 0]
    elif row[53] == 'C/C':
        feature += [0, 0, 1, 0]
    else:
        feature += [0, 0, 0, 1]    

    # Bias
    feature.append(1)

    return np.array(feature)

def extract_features(data):
    return np.array([feature_extractor(row) for row in data])

def extract_all_features(data):
    return np.array([all_feature_extractor(row) for row in data])

def calc_regret(prediction, labels, reward_function):
    regret = 0
    for row in range(len(prediction)):
        regret += 0 - reward_function(prediction[row], labels[row])
    return regret


def calc_accuracy(prediction, labels, bucket_function):
    correct = sum([prediction[row] == bucket_function(labels[row]) for row in range(len(prediction))])
    return float(correct) / len(prediction)
