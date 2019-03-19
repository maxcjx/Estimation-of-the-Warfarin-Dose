import argparse
import LASSOBandit, LinUCB, baseline
from dataset import *
import copy

def main(args):
    data, labels = load_data(args.data)
    tiny_features = extract_features(data)
    features = extract_all_features(data)

    print(np.shape(tiny_features))
    print(np.shape(features))

    
    # LinUCB
    regret_list = []
    accuracy_list = []
    num_rows = features.shape[0]

    
    for _ in range(10):
        features = copy.deepcopy(features)
        labels = copy.deepcopy(labels)
        permutation = np.random.permutation(num_rows)
        features = features[permutation]
        labels = labels[permutation]
        prediction_LinUCB = LinUCB.LinUCB(features, labels, 3, dosage_reward, 1.4)
        regret = calc_regret(prediction_LinUCB, labels, dosage_reward)
        accuracy = calc_accuracy(prediction_LinUCB, labels, dosage_bucket)
        regret_list.append(regret)
        accuracy_list.append(accuracy)
    print("-------------")
    print("LinUCB: ")
    print("Regret: mean {}, std {}".format(np.mean(regret_list), np.std(regret_list)))
    print("Accuracy: mean {}, std {}".format(np.mean(accuracy_list), np.std(accuracy_list)))
    
    '''
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
    
    for q in range(1, 3):
        h = 0.1
        while h <= 2.0:

            features = copy.deepcopy(features)
            labels = copy.deepcopy(labels)
            permutation = np.random.permutation(num_rows)
            features = features[permutation]
            labels = labels[permutation]
            prediction_LASSO = LASSOBandit.LASSOBandit(features, labels, 3, dosage_reward, 1, 1.2, 0.05, 0.05)
            regret = calc_regret(prediction_LASSO, labels, dosage_reward)
            accuracy = calc_accuracy(prediction_LASSO, labels, dosage_bucket)
            print("-------------")
            print("LASSO: q:{} h:{}".format(q, h))
            print("Regret: {}".format(regret))
            print("Accuracy: {}".format(accuracy))

            h += 0.1
    
    
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
