import pandas as pd

train_df = pd.read_csv('../data/train.txt', sep = ';', names = ['Sentence', 'Emotion'])
val_df = pd.read_csv('../data/val.txt', sep = ';', names = ['Sentence', 'Emotion'])
test_df = pd.read_csv('../data/test.txt', sep = ';', names = ['Sentence', 'Emotion'])

train_df.loc[:, 'split'] = 'train'
val_df.loc[:, 'split'] = 'val'
test_df.loc[:, 'split'] = 'test'

df = pd.concat([train_df, val_df, test_df])
df.to_csv("dataset.csv", index = False)
