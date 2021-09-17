import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
default_data_paths = 'model/vae/VAE.pkl'
# Hyper Parameter
BATCH_SIZE = 64
EPOCHS = 100
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

class Encoder(nn.Module):
    def __init__(self,input,hidden):
        super(Encoder, self).__init__()
        self.fc1_1 = nn.Linear(input, hidden)
        self.fc1_2 = nn.Linear(input, hidden)
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
    def __init__(self,input,hidden):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(hidden, input)
        self.simoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.simoid(out)

        return out

def loss_function(recon_x, x, mu, log_var):
    reconstruction_function = nn.MSELoss(size_average=False)
    MSE = reconstruction_function(recon_x, x)
    KLD_element = mu.pow(2).add_(log_var.exp()).mul_(-1).add_(1).add_(log_var)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    return MSE + KLD

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

def load_data(data_path=default_data_paths):
    df = pd.read_pickle(data_path)
    print("VAE data loaded")
    return df

def save_data(df,data_path=default_data_paths):
    df.to_pickle(data_path)
    print("VAE data saved")

def recommand(user_id):
    df = load_data()
    try:
        la = df.xs(user_id,level=0,drop_level=False).sum()
        dict_la=la.to_dict()
    except:
        dict_la=dict()
    return dict_la



