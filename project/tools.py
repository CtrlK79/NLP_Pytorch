import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter

class Vocabulary(object):
    def __init__(self, token_to_idx = None, add_unk = True, unk_token = '<UNK>'):
        if token_to_idx is None:
            token_to_idx = {}

        self.token_to_idx = token_to_idx
        self.add_unk = add_unk
        self.unk_token = unk_token

        if self.add_unk:
            self.token_to_idx[self.unk_token] = 0
        self.idx_to_token = {idx : token for token, idx in self.token_to_idx.items()}

    def add_token(self, token):
        if token in self.token_to_idx:
            index = self.token_to_idx[token]
        else:
            index = len(self.token_to_idx)
            self.token_to_idx[token] = index
            self.idx_to_token[index] = token
        return index

    def add_many(self, tokens):
        return [self.add_token(token) for token in tokens.split(" ")]

    def lookup_token(self, token):
        return self.token_to_idx[token]

    def lookup_index(self, idx):
        return self.idx_to_token[idx]

    def to_serializable(self):
        return {
        'token_to_idx' : self.token_to_idx,
        'add_unk' : self.add_unk,
        'unk_token' : self.unk_token
        }

    @classmethod
    def from_serializable(cls, serializable):
        return cls(**serializable)

#Debugging
#voc = Vocabulary()
#voc.add_token("you")
#voc.add_many('i love nlp study club')
#print(voc.idx_to_token)

class Vectorizer(object):
    def __init__(self, sentences_vocab, emotions_vocab):
        self.sentences_vocab = sentences_vocab
        self.emotions_vocab = emotions_vocab

    @classmethod
    def from_dataframe(cls, df, count = 3):
        train_dataset = df.loc[df['split']=='train', :]

        sentences_vocab = Vocabulary()
        emotions_vocab = Vocabulary(add_unk = False)

        for emotion in sorted(set(train_dataset.Emotion)):
            emotions_vocab.add_token(emotion)

        counter = Counter()
        for sentence in train_dataset.Sentence:
            for token in sentence.split(" "):
                counter[token] += 1

        for token, num in counter.items():
            if num >= count:
                sentences_vocab.add_token(token)

        return cls(sentences_vocab, emotions_vocab)

    def vectorize(self, sentence):
        one_hot = np.zeros(len(self.sentences_vocab.token_to_idx), dtype = np.float32)

        for word in sentence.split(" "):
            if word in self.sentences_vocab.token_to_idx:
                one_hot[self.sentences_vocab.lookup_token(word)] = 1
            else:
                one_hot[self.sentences_vocab.lookup_token(self.sentences_vocab.unk_token)] = 1

        return one_hot

    def to_serializable(self):
        return {
        'sentences_vocab' : self.sentences_vocab,
        'emotions_vocab' : self.emotions_vocab
        }

    @classmethod
    def from_serializable(cls, serializable):
        return cls(**serializable)

#Debugging
#df = pd.read_csv('../data/train.txt', sep = ';', names = ['Sentence', 'Emotion'])
#vec = Vectorizer.from_dataframe(df)
#print(vec.sentences_vocab.token_to_idx)
#v = vec.vectorize('do you have any specific plan to do that')
#print(v.shape)
#print(np.where(v!=0))
#for idx in np.where(v!=0)[0]:
#    print('Index:', idx, ', Word: ', vec.sentences_vocab.idx_to_token[idx])

class dataset(Dataset):
    def __init__(self, df, split = 'train'):
        self.split = split
        self.df = df
        self.vectorizer = Vectorizer.from_dataframe(self.df.loc[df['split']=='train', :])
        self.target_df = self.df.loc[self.df['split']==split, :]

    def select_split(self, split):
        self.target_df = self.df.loc[self.df['split']==split, :]

    def __len__(self):
        return len(self.target_df)

    def __getitem__(self, idx):
        return {
        'x_data' : self.vectorizer.vectorize(self.target_df.iloc[idx, 0]),
        'y_target' : self.vectorizer.emotions_vocab.token_to_idx[self.target_df.iloc[idx, 1]]
        }

#Debugging
#df = pd.read_csv('dataset.csv')
#dataset_df = dataset(df = df, split = 'test')
#dataloader_df = DataLoader(dataset_df, batch_size = 64, drop_last = False)
#for data in dataloader_df:
#    print('x_data: ', data['x_data'], ', shape: ', data['x_data'].shape)
#    print('y_target: ', data['y_target'], ', shape: ', data['y_target'].shape)
#    print('1 iter ends')
