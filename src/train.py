#!/usr/bin/env python3

import argparse
import csv
import numpy as np
from sklearn.metrics import f1_score
import xgboost as xgb
import json


def train(args):
    train = np.loadtxt('./dataset/train.csv', delimiter=',')
    val = np.loadtxt('./dataset/val.csv', delimiter=',')

    train_X = train[:, :33]
    train_Y = train[:, 34]

    val_X = val[:, :33]
    val_Y = val[:, 34]

    xg_train = xgb.DMatrix(train_X, label=train_Y)
    xg_val = xgb.DMatrix(val_X, label=val_Y)

    param = {}
    param['objective'] = 'multi:softprob'
    param['nthread'] = 1
    param['num_class'] = 6
    param['eta'] = args.learning_rate
    param['max_depth'] = args.max_depth

    num_round = args.boost_rounds
    watchlist = [(xg_train, 'train'), (xg_val, 'val')]
    results={}
    bst = xgb.train(param, xg_train, num_round, evals=watchlist, evals_result=results)
    
    pred_prob = bst.predict(xg_val)
    pred = pred_prob.argmax(axis=1)

    metrics = {
        "train" : {
            'f1' : f1_score(val_Y, pred, average='macro'),
            'accuracy' : np.sum(pred == val_Y) / val_Y.shape[0]
        }
    }

    with open('metrics/train_metrics.json', 'w') as outfile:
        outfile.write(json.dumps(metrics, indent=2) + '\n')

    with open('plots/learning_curve.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['round', 'mlogloss'])
        [writer.writerow([i, loss]) for i,loss in enumerate(results['val']['mlogloss'])]

    bst.save_model('models/model.bin')
        
if __name__ == '__main__':
    argparser=argparse.ArgumentParser("Skin disease classifier")
    argparser.add_argument('boost_rounds', type=int)
    argparser.add_argument('learning_rate', type=float)
    argparser.add_argument('max_depth', type=int)

    args = argparser.parse_args()

    train(args)
