import pandas as pd
from database import db_execute
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
default_paths = 'data/'
# Hyper Parameter
BATCH_SIZE = 64
EPOCHS = 100
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')


class CustomDataset(Dataset):

    # 데이터 정의
    def __init__(self, x_data, y_data=None):
        self.x_data = x_data
        self.y_data = y_data

    # 이 데이터 셋의 총 데이터 수
    def __len__(self):
        return len(self.x_data)

    # 어떠한 idx를 받았을 때 그에 맞는 데이터를 반환
    def __getitem__(self, idx):
        if self.y_data is None:
            x = torch.FloatTensor(self.x_data[idx])
            return x
        else:
            x = torch.FloatTensor(self.x_data[idx])
            y = torch.FloatTensor(self.y_data[idx])
            return x, y


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.fc1_1 = nn.Linear(2798, 256)
        self.fc1_2 = nn.Linear(2798, 256)
        self.relu = nn.ReLU()

    def encode(self, x):
        mu = self.relu(self.fc1_1(x))
        log_var = self.relu(self.fc1_2(x))

        return mu, log_var

    def reparametrize(self, mu, log_var):
        std = log_var.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_().to(DEVICE)

        return eps.mul(std).add_(mu)

    def forward(self, x):
        mu, log_var = self.encode(x)
        reparam = self.reparametrize(mu, log_var)

        return mu, log_var, reparam


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(256, 2798)
        self.simoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.simoid(out)

        return out


def loadData(movie_paths=default_paths):
    movie = pd.read_csv(movie_paths + "ratings.csv")
    meta = pd.read_csv(movie_paths + 'movies_metadata.csv', low_memory=False)
    meta = meta.rename(columns={'id': 'movieId'})

    movie['movieId'] = movie['movieId'].astype(str)
    meta['movieId'] = meta['movieId'].astype(str)

    movie = pd.merge(movie, meta[['movieId', 'original_title']], on='movieId')
    movie['one'] = 1
    df = movie.pivot_table(
        index='userId', columns='original_title', values='one').fillna(0)
    return df


def loss_function(recon_x, x, mu, log_var):
    MSE = reconstruction_function(recon_x, x)
    KLD_element = mu.pow(2).add_(log_var.exp()).mul_(-1).add_(1).add_(log_var)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    return MSE + KLD


def train(encoder, decoder, train_loader):
    encoder.train()
    decoder.train()
    train_loss = 0

    for feature in train_loader:

        feature = feature.to(DEVICE)
        optimizer.zero_grad()
        mu, log_var, reparam = encoder(feature)
        output = decoder(reparam)
        loss = loss_function(output, feature, mu, log_var)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)
    return train_loss


def evaluate(encoder, decoder, train_loader):
    encoder.eval()
    decoder.eval()
    result = []

    with torch.no_grad():

        for feature in train_loader:
            feature = feature.to(DEVICE)
            mu, log_var, reparam = encoder(feature)
            output = decoder(reparam)
            result.append(output.cpu().numpy())

    result = np.concatenate(result)
    return result


df = loadData(default_paths)
w_metrix = df.iloc[:, :].values
decoder = Decoder().to(DEVICE)
encoder = Encoder().to(DEVICE)
reconstruction_function = nn.MSELoss(size_average=False)
parameters = list(encoder.parameters()) + list(decoder.parameters())
optimizer = torch.optim.Adam(parameters, lr=0.0005)
train_dataset = CustomDataset(w_metrix)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=False)

for epoch in range(1, EPOCHS + 1):
    train_loss = train(encoder, decoder, train_loader)
result = evaluate(encoder, decoder, train_loader)
print(result)
