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
        self.fc1 = nn.Linear(121, 256) # state 117 + action 2
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


steps = 60000
rewards, oracle = [0] * steps, [0] * steps
max_training_steps = 50000
train_test_split = 0.9

if __name__ == '__main__':
    df = pd.read_csv('mushroom_dataset.csv', header=None)

    df['target'] = 0
    df['action1'] = 0
    df['action2'] = 0
    df['action3'] = 0
    df['action4'] = 0
    df.loc[df[0] == 'e', 'target'] = 1
    df = df.drop(0, axis=1)
    df = pd.get_dummies(df)
    # print(df.head())
    print(df.shape)

    train_test_split_idx = int(df.shape[0] * train_test_split)
    train_df = df[:train_test_split_idx]
    test_df = df[train_test_split_idx:]
    del df

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    neural_bandit = ContextualNeuralBandit().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(neural_bandit.parameters(), lr=0.0003)

    batch_transitions = pd.DataFrame()
    for i in range(steps):
        if i % 500 == 1:
            print(i)
        
        if i < max_training_steps:
            row = train_df.sample(1)
        else:
            row = test_df.sample(1)

        # row = df.iloc[[i % df.shape[0]]]
        t = row['target']
        row = row.drop('target', axis='columns')
        row1 = deepcopy(row)
        row2 = deepcopy(row)
        row3 = deepcopy(row)
        row4 = deepcopy(row)

        row1['action1'] = 1
        row2['action2'] = 1
        row3['action3'] = 1
        row4['action4'] = 1
        
        rows = [row1, row2, row3, row4]
        s_a_1234 = pd.concat(rows)
        s_a_1234 = torch.tensor(s_a_1234.values, dtype=torch.float).to(device)
        neural_bandit.eval()
        pred_rewards = neural_bandit(s_a_1234)
        _, chosen_a = torch.max(pred_rewards.data, 0)
        
        ran = random.uniform(0, 1)
        
        # if safe
        if t.values[0] == 1:
            if ran > .3:
                oracle[i] = 1 # best reward possible (avg = 0.7)

            # if action1 'cook & eat'
            if chosen_a == 0: # 0.7 avg r
                if ran > .3:
                    rows[chosen_a]['reward'] = 1
                    rewards[i] = 1
                else:
                    rows[chosen_a]['reward'] = 0
            # if action2 'raw & eat'
            elif chosen_a == 1: # 0.4 avg r
                if ran > .6:
                    rows[chosen_a]['reward'] = 1
                    rewards[i] = 1
                else:
                    rows[chosen_a]['reward'] = 0
            # if action3 'keep for now'
            elif chosen_a == 2: # 0.1 avg r
                if ran > .8:
                    rows[chosen_a]['reward'] = 1
                    rewards[i] = 1
                else:
                    rows[chosen_a]['reward'] = 0
            # if action4 'donot eat & discard'
            elif chosen_a == 3:
                rewards[i] = 0
                rows[chosen_a]['reward'] = 0

            

        # if poisonous
        elif t.values[0] == 0:
            if ran > .5:
                oracle[i] = 1

           # if action1 'cook & eat'
            if chosen_a == 0: # 0.7 avg r
                if ran > .7:
                    rows[chosen_a]['reward'] = 1
                    rewards[i] = 1
                else:
                    rows[chosen_a]['reward'] = 0
            # if action2 'raw & eat'
            elif chosen_a == 1: # 0.4 avg r
                rewards[i] = 0
                rows[chosen_a]['reward'] = 0
            # if action3 'keep for now'
            elif chosen_a == 2: # 0.1 avg r
                if ran > .9:
                    rows[chosen_a]['reward'] = 1
                    rewards[i] = 1
                else:
                    rows[chosen_a]['reward'] = 0
            # if action4 'donot eat & discard'
            elif chosen_a == 3:
                if ran > .5:
                    rows[chosen_a]['reward'] = 1
                    rewards[i] = 1
                else:
                    rows[chosen_a]['reward'] = 0

        if i < max_training_steps:
            batch_transitions = pd.concat([batch_transitions, rows[chosen_a]])


        if i < max_training_steps and batch_transitions.shape[0] == 16:
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
