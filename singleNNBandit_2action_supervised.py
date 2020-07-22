import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils

import numpy as np
import pandas as pd
import random
import copy
from copy import deepcopy
import seaborn as sns


class ContextualNeuralBandit(nn.Module):
    def __init__(self):
        super(ContextualNeuralBandit, self).__init__()
        self.fc1 = nn.Linear(118, 256) # state 117 + action 1
        self.fc2 = nn.Linear(256, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


# steps = 50000
# rewards, oracle, ps = [0] * steps, [0] * steps, [0] * steps

if __name__ == '__main__':
    df = pd.read_csv('mushroom_dataset.csv', header=None)

    df['target'] = 0
    df['action'] = 0
    df['reward'] = 0
    df.loc[df[0] == 'e', 'target'] = 1
    df = df.drop(0, axis=1)
    df = pd.get_dummies(df)
    # print(df.head())
    print(df.shape)


    for i, row in df.iterrows():
        randp = random.uniform(0, 1)
        if randp > 0.5:
            df.at[i, 'action'] = 1
        
        if row['target'] == 1:
            if row['action'] == 1:
                # randp2 = random.uniform(0, 1)
                # if randp2 > 0.4: # 60% chance of get 1 reward with a1
                df.at[i, 'reward'] = 1
        else:
            randp3 = random.uniform(0, 1)
            if row['action'] == 0:
                # if randp3 > 0.1: # 90% of get 1 reward with a0
                df.at[i, 'reward'] = 1



    Y, X = df.reward.values, df.drop(['reward', 'target'], axis='columns').values
    Y, X = np.array(Y), np.array(X)
    print(np.shape(Y))
    print(np.shape(X))
    del df

    train_test_split_idx = 6000
    X_train, Y_train = X[:train_test_split_idx], Y[:train_test_split_idx]
    X_test, Y_test = X[train_test_split_idx:], Y[train_test_split_idx:]

    train_dataset = data_utils.TensorDataset(torch.from_numpy(X_train).type(torch.FloatTensor), torch.from_numpy(Y_train).type(torch.FloatTensor))
    train_dataloader = data_utils.DataLoader(train_dataset, 64, shuffle=True, num_workers=1)

    val_dataset = data_utils.TensorDataset(torch.from_numpy(X_test).type(torch.FloatTensor), torch.from_numpy(Y_test).type(torch.FloatTensor))
    val_dataloader = data_utils.DataLoader(val_dataset, 64, shuffle=True, num_workers=1)

    dataloaders = {}
    dataloaders['train'] = train_dataloader
    dataloaders['val'] = val_dataloader

    dataset_sizes = {'train': len(Y_train), 'val': len(Y_test)}

    num_epochs = 100
    criterion = nn.BCELoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    neural_bandit = ContextualNeuralBandit().to(device)
    optimizer = optim.Adam(neural_bandit.parameters(), lr=0.001, weight_decay=0.05)


    def train_model(model):
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        preds = torch.where(outputs > 0.5, torch.ones(outputs.size()).to(device), torch.zeros(outputs.size()).to(device))

                        labels = labels.unsqueeze(1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.float() / dataset_sizes[phase]

                print('{} Loss: {:.3f} Acc: {:.3f}'.format(phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

            print()
        
        model.load_state_dict(best_model_wts)
        return model


    final_model = train_model(neural_bandit)
    torch.save(final_model, "./mushroombandit/123.pt")