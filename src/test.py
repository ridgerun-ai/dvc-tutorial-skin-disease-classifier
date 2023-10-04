#!/usr/bin/env python3

import argparse
import csv
import json
import numpy as np
import xgboost as xgb
from sklearn.metrics import f1_score

def test(args):
    bst = xgb.Booster()

    bst.load_model('models/model.bin')

    test = np.loadtxt('./dataset/test.csv', delimiter=',')

    test_X = test[:, :33]
    test_Y = test[:, 34]

    xg_test = xgb.DMatrix(test_X, label=test_Y)

    pred_prob = bst.predict(xg_test)
    pred_max = np.max(pred_prob, axis=1)
    pred = pred_prob.argmax(axis=1)

    # Discard the ambiguous results (prob below our thershold)
    pred[pred_max < args.threshold] = -1

    metrics = {
        "test" : {
            'f1' : f1_score(test_Y, pred, average='macro'),
            'accuracy' : np.sum(pred == test_Y) / test_Y.shape[0]
        }
    }

    with open('metrics/test_metrics.json', 'w') as outfile:
        outfile.write(json.dumps(metrics, indent=2) + '\n')

    with open('plots/confusion_matrix.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['actual', 'predicted'])
        [writer.writerow([int(test_Y[i]), pred[i]]) for i in range(pred.shape[0])]

if __name__ == '__main__':
    argparser=argparse.ArgumentParser("Skin disease classifier tester")
    argparser.add_argument('threshold', type=float)
    args = argparser.parse_args() 

    test(args)
