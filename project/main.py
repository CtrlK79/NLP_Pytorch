import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
from tqdm import tqdm

from classifiers import FCN_1, FCN_2, FCN_3, FCN_4
from tools import Vocabulary, Vectorizer, dataset

# Load dataset
df = pd.read_csv('dataset.csv')
dataset_df = dataset(df)

# case flag
case = 2

if case == 1:
    model = FCN_1(input_features = len(dataset_df.vectorizer.sentences_vocab.token_to_idx), output_features = len(dataset_df.vectorizer.emotions_vocab.token_to_idx))
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr = 0.01)
    epochs = 5

elif case == 2:
    model = FCN_2(input_features = len(dataset_df.vectorizer.sentences_vocab.token_to_idx), output_features = len(dataset_df.vectorizer.emotions_vocab.token_to_idx))
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr = 0.01)
    epochs = 5

elif case == 3:
    model = FCN_3(input_features = len(dataset_df.vectorizer.sentences_vocab.token_to_idx), output_features = len(dataset_df.vectorizer.emotions_vocab.token_to_idx))
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr = 0.001)
    epochs = 10

elif case == 4:
    model = FCN_4(input_features = len(dataset_df.vectorizer.sentences_vocab.token_to_idx), output_features = len(dataset_df.vectorizer.emotions_vocab.token_to_idx))
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr = 0.001)
    epochs = 10

print("----------model #{} start!----------".format(case))
batch_size = 128
running_loss = 0.0
running_acc = 0.0

try:
    with tqdm(total = epochs, desc = 'epochs') as epoch_bar:
        for epoch in range(epochs):

            dataset_df.select_split('train')
            dataloader = DataLoader(dataset_df, batch_size = batch_size, drop_last = True)

            running_loss = 0.0
            running_acc = 0.0
            loop = 0
            model.train()

            with tqdm(desc = 'training batch', total = len(dataset_df.target_df) // batch_size) as batch_training_bar:
                for batch in dataloader:
                    loop += 1

                    optim.zero_grad()

                    y_pred = model(batch['x_data'])

                    loss = loss_fn(y_pred, batch['y_target'])
                    acc = int((y_pred.max(axis = 1)[1] == batch['y_target']).sum()) / len(batch['y_target'])

                    loss.backward()

                    optim.step()

                    running_loss = running_loss + (loss - running_loss) / loop
                    running_acc = running_acc + (acc - running_acc) / loop

                    batch_training_bar.set_postfix(loss = float(running_loss), acc = float(running_acc), epoch = epoch)
                    batch_training_bar.update()

                dataset_df.select_split('val')
                dataloader = DataLoader(dataset_df, batch_size = batch_size, drop_last = True)

                running_loss = 0.0
                running_acc = 0.0
                loop = 0
                model.eval()

            with tqdm(desc = 'val batch', total = len(dataset_df.target_df) // batch_size) as batch_val_bar:
                for batch in dataloader:
                    loop += 1

                    y_pred = model(batch['x_data'])

                    loss = loss_fn(y_pred, batch['y_target'])
                    acc = int((y_pred.max(axis = 1)[1] == batch['y_target']).sum()) / len(batch['y_target'])

                    running_loss = running_loss + (loss - running_loss) / loop
                    running_acc = running_acc + (acc - running_acc) / loop

                    batch_val_bar.set_postfix(loss = float(running_loss), acc = float(running_acc), epoch = epoch)
                    batch_val_bar.update()

            epoch_bar.update()

except KeyboardInterrupt:
    print("Exiting loop")


epoch_bar.clear()
batch_training_bar.clear()
batch_val_bar.clear()


# test

dataset_df.select_split('test')
model.eval()
test_loss = 0.0
test_acc = 0.0

dataloader = DataLoader(dataset_df, batch_size = 2000)
for batch in dataloader:
    y_pred = model(batch['x_data'])
    test_loss = loss_fn(y_pred, batch['y_target'])
    test_acc = int((y_pred.max(axis = 1)[1] == batch['y_target']).sum()) / len(batch['y_target'])

print(model)
print('Test Loss: {loss}'.format(loss = test_loss))
print('Test Accuracy: {acc}'.format(acc = test_acc))
print('----------finished----------')
