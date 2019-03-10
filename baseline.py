import numpy as np
from dataset import *

def fixed_dose(features, labels):
    num_rows = features.shape[0]
    prediction = []
    for row in range(num_rows):
        dosage = 5 * 7
        prediction.append(dosage_bucket(dosage))
    return prediction


def Warfarin_clinical_dosing(features, labels):
    num_rows = features.shape[0]
    prediction = []
    # w = np.array([-0.2546, 0.0118, 0.0134, -0.6752, 0.4060, 0.0443, 1.2799, -0.5695])
    # b = 4.0376
    w = np.array([-0.2546, 0.0118, 0.0134, -0.6752, 0.4060, 0.0443, 1.2799, -0.5695, 4.0376])
    for row in range(num_rows):
        dosage = (np.dot(w, features[row])) ** 2
        prediction.append(dosage_bucket(dosage))
    return prediction