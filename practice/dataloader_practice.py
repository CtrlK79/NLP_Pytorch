import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class dataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx): # must return tensors, numpy arrays, numbers, dicts or lists
        return [self.df.iloc[idx, 0], self.df.iloc[idx, 1]]

df_ex = pd.read_csv('data/val.txt', sep = ';', names = ['Sentence', 'Emotion'])
dataset_ex = dataset(df_ex)
dataloader_ex = DataLoader(dataset_ex, batch_size = 10, drop_last = False)

for row in dataloader_ex: # row : list of tuple(=[(), ()]) and the size of the tuples are same with batch_size
    print('Sentence: ', row[0])
    print('Emotion: ', row[1])
    print('Type: ', type(row)) # list, it is governed by __getitem__()
    print('Type of row[0]', type(row[0])) # tuple
    print('-----------------------')
    print('1 iter ends')
    print('-----------------------')
