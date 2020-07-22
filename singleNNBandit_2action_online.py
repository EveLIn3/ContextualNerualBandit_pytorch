import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import random
import copy
from copy import deepcopy
import seaborn as sns


class ContextualNeuralBandit(nn.Module):
    def __init__(self):
        super(ContextualNeuralBandit, self).__init__()
        self.fc1 = nn.Linear(119, 256) # state 117 + action 2
        self.fc3 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 1)
        self.dropout1 = nn.Dropout(p=0.1) # Dropout is used as exploration
        self.dropout2 = nn.Dropout(p=0.1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc3(x))
        x = self.dropout2(x)
        x = torch.sigmoid(self.fc2(x))
        return x


steps = 25000
rewards, oracle = [0] * steps, [0] * steps
max_training_steps = 10000

if __name__ == '__main__':
    df = pd.read_csv('mushroom_dataset.csv', header=None)

    df['target'] = 0
    df['action1'] = 0
    df['action2'] = 0
    df.loc[df[0] == 'e', 'target'] = 1
    df = df.drop(0, axis=1)
    df = pd.get_dummies(df)
    # print(df.head())
    print(df.shape)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    neural_bandit = ContextualNeuralBandit().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(neural_bandit.parameters(), lr=0.0003)

    train_df = df[:7000]
    test_df = df[7000:]

    del df

    batch_transitions = pd.DataFrame()
    for i in range(steps):
        if (i % 500 == 1):
            print(i)
        if i < max_training_steps:
            row = train_df.sample(1)
        else: # stop training after this point
            # row = test_df.iloc[[i % test_df.shape[0]]]
            row = test_df.sample(1)
        t = row['target']
        row = row.drop('target', axis='columns')
        row1 = deepcopy(row)
        row2 = deepcopy(row)

        row1['action1'] = 1
        row2['action2'] = 1

        x_with_a1 = torch.tensor(row1.values, dtype=torch.float).to(device)
        x_with_a2 = torch.tensor(row2.values, dtype=torch.float).to(device)

        neural_bandit.eval()
        a1 = neural_bandit(x_with_a1)
        a2 = neural_bandit(x_with_a2)
        
        ran = random.uniform(0, 1)
        
        # if safe
        if t.values[0] == 1:
            oracle[i] = 1 # best reward possible
            
            # if eat
            if a1 >= a2:
                row1['reward'] = 1
                rewards[i] = 1
                batch_transitions = pd.concat([batch_transitions, row1])

            # if dont eat
            elif a1 < a2:
                if ran > .5:
                    rewards[i] = 1
                    row2['reward'] = 1
                else:
                    row2['reward'] = 0
                batch_transitions = pd.concat([batch_transitions, row2])
                
        # if poisonous
        elif t.values[0] == 0:
            if ran > .5:
                oracle[i] = 1

            # if eat
            if a1 >= a2:
                row1['reward'] = 0
                batch_transitions = pd.concat([batch_transitions, row1])
            # if dont eat
            elif a1 < a2:
                if ran > .5:
                    rewards[i] = 1
                    row2['reward'] = 1
                else:
                    row2['reward'] = 0
                
                batch_transitions = pd.concat([batch_transitions, row2])
        
        if i < max_training_steps and batch_transitions.shape[0] == 4:
            Y = torch.tensor(batch_transitions.reward.values, dtype=torch.float).to(device)
            X = torch.tensor(batch_transitions.drop('reward', axis='columns').values, dtype=torch.float).to(device)

            neural_bandit.train()
            pred = neural_bandit(X)
            label = Y.unsqueeze(1)
            
            loss = criterion(pred, label)
            # print("Loss: {:.4f}".format(loss.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_transitions = pd.DataFrame()


    cumu_reward = pd.Series(rewards).cumsum()
    oracle_reward = pd.Series(oracle).cumsum()
    cumu_regret = oracle_reward - cumu_reward

    sns.lineplot(x=range(steps), y=cumu_regret)
    plt.show()
