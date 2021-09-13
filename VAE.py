import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from model.database import db_execute

default_data_paths = 'model/vae/VAE.pkl'
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
        self.fc1_1 = nn.Linear(13, 2)
        self.fc1_2 = nn.Linear(13, 2)
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
        self.fc1 = nn.Linear(2, 13)
        self.simoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.simoid(out)

        return out


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
def get_click_comment():
    # clicks frame
    sql = """select * from clicks where deletedAt is null"""
    re = db_execute(sql)
    cf = pd.DataFrame(re)
    cf['createdAt'] = cf['createdAt'].dt.strftime('%y-%m-%d')
    cf['one'] = 1
    cf = cf.groupby(['userId', 'createdAt', 
    'diaryId', 'emotionType', 'emotionLevel'])['one'].sum().reset_index()
    cf_emotion = pd.pivot_table(cf, index=['userId', 'createdAt'],
                                columns=['emotionType'], values='emotionLevel')
    cf_diary = pd.pivot_table(cf, index=['userId', 'createdAt'],
                              columns=['diaryId'], values='one')
    cf = pd.merge(cf_emotion, cf_diary, 'left', on=['userId', 'createdAt'])
    # comments frame
    sql = "select * from comments where deletedAt is null"
    result = db_execute(sql)
    dq = pd.DataFrame(result)
    dq['createdAt'] = dq['updatedAt'].dt.strftime('%y-%m-%d')
    dq_emotion = pd.pivot_table(dq, index=['userId', 'createdAt'],
                                columns=['userEmotionType'], values='userEmotionLevel')
    dq_diary = pd.pivot_table(dq, index=['userId', 'createdAt'],
                              columns=['diaryId'], values='emotionLevel')
    dq = pd.merge(dq_emotion, dq_diary, 'left', on=['userId', 'createdAt'])
    da = pd.concat([cf, dq]).fillna(0).sort_index()
    return da

# df = get_click_comment()
# w_metrix = df.iloc[:, :].values
# encoder = Encoder().to(DEVICE)
# decoder = Decoder().to(DEVICE)
# reconstruction_function = nn.MSELoss(size_average=False)
# parameters = list(encoder.parameters()) + list(decoder.parameters())
# optimizer = torch.optim.Adam(parameters, lr=0.0005)
# train_dataset = CustomDataset(w_metrix)

# train_loader = DataLoader(
#     train_dataset,
#     batch_size=BATCH_SIZE,
#     shuffle=False,
#     drop_last=False)

# for epoch in range(1, EPOCHS + 1):
#     train_loss = train(encoder, decoder, train_loader)
# result = evaluate(encoder, decoder, train_loader)

# df = pd.DataFrame(result,index=df.index,columns=df.columns)
# save_data(df)


