from VAE import loss_function

class Trainer():

    def __init__(self, encoder, decoder,optimizer,DEVICE):
        self.encoder = encoder
        self.decoder=decoder
        self.optimizer = optimizer
        self.loss_function=loss_function
        self.DEVICE = DEVICE

        super().__init__()

    def train(self, train_loader,EPOCHS):
        self.encoder.train()
        self.decoder.train()
        lowest_loss= float("inf")
        best_encoder=None
        best_decoder=None 
        for epoch in range(1, EPOCHS + 1):
            train_loss = 0
            for feature in train_loader:

                feature = feature.to(self.DEVICE)
                self.optimizer.zero_grad()
                mu, log_var, reparam = self.encoder(feature)
                output = self.decoder(reparam)
                loss = self.loss_function(output, feature, mu, log_var)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)
            # best model saved
            if train_loss <= lowest_loss:
                lowest_loss = train_loss
                best_encoder = self.encoder
                best_decoder = self.decoder

        if best_encoder is None:
            return self.encoder , self.decoder
        else:
            print(f"\nTrain Loss: {lowest_loss:.4f}")
            return best_encoder, best_decoder
