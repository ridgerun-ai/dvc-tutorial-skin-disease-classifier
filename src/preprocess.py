#!/usr/bin/env python3

import numpy as np
import pandas as pd

np.random.seed(8)

# 1. Load data from a CSV file
df = pd.read_csv('dataset/features.csv', header=None)

# 2. Read values from the 33rd column and replace with 1 if the value is a '?' char or 0 otherwise
df.iloc[:, 33] = df.iloc[:, 33].apply(lambda x: int(x == '?'))

# 3. Read values from the 34th column and decrease by 1
df.iloc[:, 34] = df.iloc[:, 34].astype(int) - 1

# 4. Take 80%, 10% and 10% for train, val and test splits
train, val, test = np.split(df.sample(frac=1), [int(.8*len(df)), int(.9*len(df))])

# 5. Save the splits into their	respective files
train.to_csv('dataset/train.csv', index=False, header=None)
val.to_csv('dataset/val.csv', index=False, header=None)
test.to_csv('dataset/test.csv', index=False, header=None)
