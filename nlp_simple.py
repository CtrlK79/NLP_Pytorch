import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
from tqdm import tqdm

class Vocabulary():
    def __init__(self, data = None, add_unk = True, unk_token = '<UNK>'):
        self.idx_to_token = {}
        self.token_to_idx = {}
        self.add_unk = add_unk
        self.unk_token = unk_token
        self.data = data

        if self.add_unk:
            self.add_token(unk_token)

        if data != None:
            self.add_many(data)

    def add_token(self, token):
        if token in self.token_to_idx:
            return self.token_to_idx[token]
        else:
            self.token_to_idx[token] = len(self.idx_to_token)
            self.idx_to_token[len(self.idx_to_token)] = token
            return self.token_to_idx[token]

    def add_many(self, tokens):
        for token in tokens.split(" "):
            self.add_token(token)

#Vocabulary class verification code:
#voc = Vocabulary("hi my name is jaewoo nice to meet")
#voc.add_token("you")
#voc.add_many('i love nlp study club')
#print(voc.idx_to_token)

class Vectorizer():
    def __init__(self, sentences_vocab = Vocabulary(), emotions_vocab = Vocabulary(add_unk = False)):
        self.sentences_vocab = sentences_vocab
        self.emotions_vocab = emotions_vocab

    @classmethod
    def from_dataframe(cls, df):
        sentences_vocab = Vocabulary()
        emotions_vocab = Vocabulary(add_unk = False)

        for data in df.loc[:, 'Sentence']:
            sentences_vocab.add_many(data)

        for data in df.loc[:, 'Emotion']:
            emotions_vocab.add_many(data)

        return cls(sentences_vocab = sentences_vocab, emotions_vocab = emotions_vocab)

    def vectorize(self, sentence):
        one_hot = np.zeros(len(self.sentences_vocab.token_to_idx), dtype = np.float32)

        for word in sentence.split(" "):
            if word in self.sentences_vocab.token_to_idx:
                one_hot[self.sentences_vocab.token_to_idx[word]] = 1
            else:
                one_hot[self.sentences_vocab.token_to_idx[self.sentences_vocab.unk_token]] = 1

        return one_hot

    #def emotions_classification(self, emotion):
    #    one_hot = np.zeros(len(self.emotions_vocab.token_to_idx), dtype = np.float32)
    #    one_hot[self.emotions_vocab.token_to_idx[emotion]] = 1

    #    return one_hot

#Vectorizer class verification
#df = pd.read_csv('data/test.txt', sep = ';', names = ['Sentence', 'Emotion'])
#vec = Vectorizer.from_dataframe(df)
#v = vec.vectorize('do you have any specific plan to do that')
#print(v.shape)
#print(np.where(v!=0))
#for idx in np.where(v!=0)[0]:
#    print('Index:', idx, ', Word: ', vec.sentences_vocab.idx_to_token[idx])

class dataset(Dataset):
    def __init__(self, df, split = 'train', train_vectorizer = Vectorizer()):
        self.df = df
        self.split = split
        self.vectorizer = Vectorizer.from_dataframe(self.df)
        self.train_vectorizer = train_vectorizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.split=='train':
            return {
            'x_data' : self.vectorizer.vectorize(self.df.iloc[idx, 0]),
            'y_target' : self.vectorizer.emotions_vocab.token_to_idx[self.df.iloc[idx, 1]]
            }
        else:
            return {
            'x_data' : self.train_vectorizer.vectorize(self.df.iloc[idx, 0]),
            'y_target' : self.train_vectorizer.emotions_vocab.token_to_idx[self.df.iloc[idx, 1]]
            }


#dataset class verification
#df = pd.read_csv('data/test.txt', sep = ';', names = ['Sentence', 'Emotion'])
#dataset_df = dataset(df)
#dataloader_df = DataLoader(dataset_df, batch_size = 64, drop_last = False)
#for data in dataloader_df:
#    print('x_data: ', data['x_data'], ', shape: ', data['x_data'].shape)
#    print('y_target: ', data['y_target'], ', shape: ', data['y_target'].shape)
#    print('1 iter ends')

class Net(nn.Module):
    def __init__(self, input_features, output_features):
        super(Net, self).__init__()
        self.layer_1 = nn.Linear(input_features, 256)
        self.layer_2 = nn.Linear(256, 128)
        self.layer_3 = nn.Linear(128, 64)
        self.layer_4 = nn.Linear(64, 32)
        self.layer_5 = nn.Linear(32, output_features)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.layer_1(x)
        x = self.relu(x)

        x = self.layer_2(x)
        x = self.relu(x)

        x = self.layer_3(x)
        x = self.relu(x)

        x = self.layer_4(x)
        x = self.relu(x)

        x = self.layer_5(x)

        return x

#Net class verification
#net = Net(1000, 8)
#summary(net, (1, 1000), batch_size = 64)

def predict(sentence, emotion):
    print('Predicted Emotion: ', training_set.vectorizer.emotions_vocab.idx_to_token[int(net(torch.Tensor(training_set.vectorizer.vectorize(sentence))).max(axis = 0)[1])])
    print('Target Emotion: ', emotion)


if __name__ == '__main__':

    training_set = dataset(pd.read_csv('data/train.txt', sep = ';', names = ['Sentence', 'Emotion']))
    val_set = dataset(pd.read_csv('data/val.txt', sep = ';', names = ['Sentence', 'Emotion']), split = 'val', train_vectorizer = training_set.vectorizer)
    test_set = dataset(pd.read_csv('data/test.txt', sep = ';', names = ['Sentence', 'Emotion']), split = 'test', train_vectorizer = training_set.vectorizer)

    net = Net(input_features = len(training_set.vectorizer.sentences_vocab.token_to_idx), output_features = len(training_set.vectorizer.emotions_vocab.token_to_idx))
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(net.parameters(), lr = 0.01)

    epochs = 5
    batch_size = 128




    epoch_bar = tqdm(desc = 'epochs', total = epochs)
    for epoch in range(epochs):

        dataloader = DataLoader(training_set, batch_size = batch_size, drop_last = True)

        running_loss = 0.0
        running_acc = 0.0
        loop = 0
        net.train()

        batch_training_bar = tqdm(desc = 'training batch', total = len(training_set.df) // batch_size)
        for batch in dataloader:
            loop += 1

            optim.zero_grad()

            y_pred = net(batch['x_data'])

            loss = loss_fn(y_pred, batch['y_target'])
            acc = int((y_pred.max(axis = 1)[1] == batch['y_target']).sum()) / len(batch['y_target'])

            loss.backward()

            optim.step()

            running_loss = (running_loss * (loop - 1) + loss) / loop
            running_acc = (running_acc * (loop - 1) + acc)

            batch_training_bar.update(1)
            batch_training_bar.set_description("loss: {l}, acc: {a}".format(l = running_loss, a = running_acc))

        dataloader = DataLoader(val_set, batch_size = batch_size, drop_last = True)

        running_loss = 0.0
        running_acc = 0.0
        loop = 0
        net.eval()

        batch_val_bar = tqdm(desc = 'validation batch', total = len(val_set.df) // batch_size)
        for batch in dataloader:
            loop += 1

            y_pred = net(batch['x_data'])

            loss = loss_fn(y_pred, batch['y_target'])
            acc = int((y_pred.max(axis = 1)[1] == batch['y_target']).sum()) / len(batch['y_target'])

            running_loss = (running_loss * (loop-1) + loss) / loop
            running_acc = (running_acc * (loop - 1) + acc) / loop

            batch_val_bar.update(1)
            batch_val_bar.set_description("loss: {l}, acc: {a}".format(l = running_loss, a = running_acc))

        epoch_bar.update(1)

    epoch_bar.clear()
    batch_training_bar.clear()
    batch_val_bar.clear()
    torch.save(net.state_dict(), 'net.pth')

    predict('i love you hyejun', 'love')
