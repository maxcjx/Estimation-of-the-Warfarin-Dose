import argparse
import LASSOBandit, LinUCB, baseline
from dataset import *
import copy

def main(args):
    data, labels = load_data(args.data)
    tiny_features = extract_features(data)
    features = extract_features(data)
    #features = extract_all_features(data)

    
    # LinUCB
    regret_list = []
    accuracy_list = []
    medium_arm_list = []
    num_rows = features.shape[0]
    reward_function = dosage_reward

    for _ in range(10):
        features = copy.deepcopy(features)
        labels = copy.deepcopy(labels)
        permutation = np.random.permutation(num_rows)
        features = features[permutation]
        labels = labels[permutation]
        prediction_LinUCB = LinUCB.LinUCB(features, labels, 3, reward_function, 1.4)
        regret = calc_regret(prediction_LinUCB, labels, reward_function)
        accuracy = calc_accuracy(prediction_LinUCB, labels, dosage_bucket)
        medium_arm = calc_bucket(prediction_LinUCB, 1)
        regret_list.append(regret)
        accuracy_list.append(accuracy)
        medium_arm_list.append(medium_arm)
    print("-------------")
    print("LinUCB: ")
    print("Regret: mean {}, std {}".format(np.mean(regret_list), np.std(regret_list)))
    print("Accuracy: mean {}, std {}".format(np.mean(accuracy_list), np.std(accuracy_list)))
    print("Max accuracy: {}".format(max(accuracy_list)))
    print("Medium arm percentage: mean {}, std {}".format(np.mean(medium_arm_list), np.std(medium_arm_list)))


    # LinUCB with bucket difference reward (0, -1, -2)
    regret_list = []
    accuracy_list = []
    medium_arm_list = []
    num_rows = features.shape[0]
    reward_function = dosage_reward_difference

    for _ in range(10):
        features = copy.deepcopy(features)
        labels = copy.deepcopy(labels)
        permutation = np.random.permutation(num_rows)
        features = features[permutation]
        labels = labels[permutation]
        prediction_LinUCB = LinUCB.LinUCB(features, labels, 3, reward_function, 1.4)
        regret = calc_regret(prediction_LinUCB, labels, reward_function)
        accuracy = calc_accuracy(prediction_LinUCB, labels, dosage_bucket)
        medium_arm = calc_bucket(prediction_LinUCB, 1)
        regret_list.append(regret)
        accuracy_list.append(accuracy)
        medium_arm_list.append(medium_arm)
    print("-------------")
    print("LinUCB with bucket difference reward (0, -1, -2): ")
    print("Regret: mean {}, std {}".format(np.mean(regret_list), np.std(regret_list)))
    print("Accuracy: mean {}, std {}".format(np.mean(accuracy_list), np.std(accuracy_list)))
    print("Max accuracy: {}".format(max(accuracy_list)))
    print("Medium arm percentage: mean {}, std {}".format(np.mean(medium_arm_list), np.std(medium_arm_list)))

    # LinUCB with real value reward
    regret_list = []
    accuracy_list = []
    medium_arm_list = []
    num_rows = features.shape[0]
    reward_function = dosage_reward_real_value

    for _ in range(10):
        features = copy.deepcopy(features)
        labels = copy.deepcopy(labels)
        permutation = np.random.permutation(num_rows)
        features = features[permutation]
        labels = labels[permutation]
        prediction_LinUCB = LinUCB.LinUCB(features, labels, 3, reward_function, 1.4)
        regret = calc_regret(prediction_LinUCB, labels, reward_function)
        accuracy = calc_accuracy(prediction_LinUCB, labels, dosage_bucket)
        medium_arm = calc_bucket(prediction_LinUCB, 1)
        regret_list.append(regret)
        accuracy_list.append(accuracy)
        medium_arm_list.append(medium_arm)
    print("-------------")
    print("LinUCB with real value reward: ")
    print("Regret: mean {}, std {}".format(np.mean(regret_list), np.std(regret_list)))
    print("Accuracy: mean {}, std {}".format(np.mean(accuracy_list), np.std(accuracy_list)))
    print("Max accuracy: {}".format(max(accuracy_list)))
    print("Medium arm percentage: mean {}, std {}".format(np.mean(medium_arm_list), np.std(medium_arm_list)))
    
    '''
    Test Hyper parameters
    Hyper parameters with good results
    q = 1, h = 0.6
    q = 1, h = 0,9
    q = 1, h = 1.2 (best)
    q = 1, h = 1.3
    q = 1, h = 1.4 
    q = 1, h = 1.5
    q = 1, h = 1.6
    q = 1, h = 1.7
    q = 1, h = 1.8
    q = 1, h = 1.9
    q = 2, h = 0.9
    q = 2, h = 1.1
    q = 2, h = 1.8

    '''
    
    '''
    for q in range(1, 4):

        h = 0.1
        while h <= 2.0:
            regret_list = []
            accuracy_list = []
            medium_arm_list = []

            for _ in range(10):
                features = copy.deepcopy(features)
                labels = copy.deepcopy(labels)
                permutation = np.random.permutation(num_rows)
                features = features[permutation]
                labels = labels[permutation]
                prediction_LASSO = LASSOBandit.LASSOBandit(features, labels, 3, dosage_reward, q, h, 0.05, 0.05)
                regret = calc_regret(prediction_LASSO, labels, dosage_reward)
                accuracy = calc_accuracy(prediction_LASSO, labels, dosage_bucket)
                regret_list.append(regret)
                accuracy_list.append(accuracy)
                medium_arm_list.append(medium_arm)

            print("-------------")
            print("LASSO: q:{} h:{}".format(q, h))
            print("Regret: mean {}, std {}".format(np.mean(regret_list), np.std(regret_list)))
            print("Accuracy: mean {}, std {}".format(np.mean(accuracy_list), np.std(accuracy_list)))
            print("Max accuracy: {}".format(max(accuracy_list)))
            print("Medium arm percentage: mean {}, std {}".format(np.mean(medium_arm_list), np.std(medium_arm_list)))

            h += 0.1
    '''
    
    
    # Test reward functions for LASSO
    for i in range(3):
        function_type = ''
        if i == 0:
            reward_function = dosage_reward
        elif i == 1:
            reward_function = dosage_reward_difference
            function_type = ' with bucket difference reward (0, -1, -2)'
        else:
            reward_function = dosage_reward_real_value
            function_type = ' with real value reward'

        regret_list = []
        accuracy_list = []
        medium_arm_list = []

        for j in range(10):
            print("At iteration {}".format(j))
            features = copy.deepcopy(features)
            labels = copy.deepcopy(labels)
            permutation = np.random.permutation(num_rows)
            features = features[permutation]
            labels = labels[permutation]
            prediction_LASSO = LASSOBandit.LASSOBandit(features, labels, 3, reward_function, 1, 5, 0.05, 0.05)
            regret = calc_regret(prediction_LASSO, labels, dosage_reward)
            accuracy = calc_accuracy(prediction_LASSO, labels, dosage_bucket)
            medium_arm = calc_bucket(prediction_LASSO, 1)
            regret_list.append(regret)
            accuracy_list.append(accuracy)
            medium_arm_list.append(medium_arm)


        print("-------------")
        print("LASSO{}: ".format(function_type,))
        print("Regret: mean {}, std {}".format(np.mean(regret_list), np.std(regret_list)))
        print("Accuracy: mean {}, std {}".format(np.mean(accuracy_list), np.std(accuracy_list)))
        print("Max accuracy: {}".format(max(accuracy_list)))
        print("Medium arm percentage: mean {}, std {}".format(np.mean(medium_arm_list), np.std(medium_arm_list)))
    
    
    
    _, labels = load_data(args.data)
    # Fixed-dose
    prediction_fixed_dose = baseline.fixed_dose(tiny_features, labels)
    print("-------------")
    print("Fixed-dose: ")
    print("Accuracy: ", calc_accuracy(prediction_fixed_dose, labels, dosage_bucket))

    # Warfarin Clinical Dosing Algorithm
    prediction_linear = baseline.Warfarin_clinical_dosing(tiny_features, labels)
    print("-------------")
    print("Warfarin Clinical Dosing Algorithm: ")
    print("Accuracy: ", calc_accuracy(prediction_linear, labels, dosage_bucket))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='./data/warfarin.csv')
    args = parser.parse_args()
    main(args)
