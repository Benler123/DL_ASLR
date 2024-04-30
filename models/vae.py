import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, num_landmarks, num_frames, hidden_size, latent_dim):
        super(VAE, self).__init__()
        self.num_landmarks = num_landmarks
        self.num_frames = num_frames
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim

        self.lstm_enc = nn.LSTM(input_size=num_landmarks * 2, hidden_size=hidden_size, batch_first=True)
        self.fc_mu = nn.Linear(hidden_size, latent_dim)
        self.fc_var = nn.Linear(hidden_size, latent_dim)

        self.fc_dec = nn.Linear(latent_dim, hidden_size)
        self.lstm_dec = nn.LSTM(input_size=hidden_size, hidden_size=num_landmarks * 2, batch_first=True)

    def encode(self, x):
        batch_size, _, _ = x.size()
        x = x.view(batch_size, self.num_frames, -1)
        _, (h_n, _) = self.lstm_enc(x)
        h_n = h_n[-1]  
        mu = self.fc_mu(h_n)
        log_var = self.fc_var(h_n)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = self.fc_dec(z)
        z = z.unsqueeze(1).repeat(1, self.num_frames, 1)
        output, _ = self.lstm_dec(z)
        return output.view(-1, self.num_frames, self.num_landmarks * 2)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

def loss_function(recon_x, x, mu, log_var):
    BCE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD