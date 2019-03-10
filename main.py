import argparse
import LinUCB, baseline
from dataset import *
import copy

def main(args):
    data, labels = load_data(args.data)
    features = extract_features(data)

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

    # Fixed-dose
    prediction_fixed_dose = baseline.fixed_dose(features, labels)
    print("-------------")
    print("Fixed-dose: ")
    print("Accuracy: ", calc_accuracy(prediction_fixed_dose, labels, dosage_bucket))

    # Warfarin Clinical Dosing Algorithm
    prediction_linear = baseline.Warfarin_clinical_dosing(features, labels)
    print("-------------")
    print("Warfarin Clinical Dosing Algorithm: ")
    print("Accuracy: ", calc_accuracy(prediction_linear, labels, dosage_bucket))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='./data/warfarin.csv')
    args = parser.parse_args()
    main(args)
