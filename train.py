import Fasttext
import LDA
import VAE
from trainer import Trainer
from dataloader import VAEDataset ,get_click_comment
from VAE import Decoder , Encoder
from torch.utils.data import DataLoader
import torch
import pandas as pd

# Hyper Parameter
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE=0.0005
default_decoder_paths = 'model/vae/decoder.pth'
default_encoder_paths = 'model/vae/encoder.pth'

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')
def FTtrain():
    print("#----FastText----#")
    corpus = Fasttext.make_corpus()
    if corpus:
        model = Fasttext.train(corpus, True)
        Fasttext.save_model(model)
        Fasttext.make_tsv(model)


def LDAtrain():
    print("#----LDA----#")
    corpus, dictionary = LDA.make_corpus()
    if corpus:
        model, dictionary = LDA.train(corpus, dictionary, True)
        LDA.save_model(model, dictionary)

def VAEtrain():
    print("#----VAE----#")
    df = get_click_comment()
    input=df.shape[1]
    hidden=input//4
    w_metrix = df.iloc[:, :].values
    encoder = Encoder(input,hidden).to(DEVICE)
    decoder = Decoder(input,hidden).to(DEVICE)
    parameters = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(parameters, lr=LEARNING_RATE)
    train_dataset = VAEDataset(w_metrix)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False)
    trainer= Trainer(
        encoder, 
        decoder,
        optimizer,
        DEVICE)
    encoder, decoder =trainer.train(train_loader,EPOCHS)
    torch.save(
        encoder.state_dict(),
        default_encoder_paths)
    torch.save(
        decoder.state_dict(),
        default_decoder_paths)
    result = VAE.evaluate(encoder, decoder, train_loader)
    df = pd.DataFrame(result,index=df.index,columns=df.columns)
    VAE.save_data(df)   
    

