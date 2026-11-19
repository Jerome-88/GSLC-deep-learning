import math

import torch
from torch import nn
from torch.nn import functional as F


class Decoder(nn.Module):
    def __init__(self, latent=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(latent, 256), nn.ReLU(),
                                 nn.Linear(256, 784), nn.Sigmoid())

    def forward(self, z):
        return self.net(z).reshape(-1, 1, 28, 28)

class Autoencoder(nn.Module):
    def __init__(self, latent=64):
        super().__init__()
        self.encoder = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(),
                                     nn.Linear(256, latent), nn.ReLU())
        self.decoder = Decoder(latent)

    def forward(self, x):
        return self.decoder(self.encoder(x))

class Classifier(nn.Module):
    def __init__(self, latent=64):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, latent)
        self.fc3 = nn.Linear(latent, 10)

    def encode(self, x):
        return F.relu(self.fc2(F.relu(self.fc1(x.flatten(1)))))

    def forward(self, x):
        return self.fc3(self.encode(x))

    def transpose_latent(self, z):
        h = F.relu(z @ self.fc2.weight)
        return torch.sigmoid(h @ self.fc1.weight).reshape(-1, 1, 28, 28)

    def transpose_class(self, y):
        one_hot = F.one_hot(y, 10).float()
        return self.transpose_latent(F.relu(one_hot @ self.fc3.weight))

class VAE(nn.Module):
    def __init__(self, latent=64):
        super().__init__()
        self.encoder = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU())
        self.mu = nn.Linear(256, latent)
        self.logvar = nn.Linear(256, latent)
        self.decoder = Decoder(latent)

    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return self.decoder(z), mu, logvar

class GAN(nn.Module):
    def __init__(self, latent=100, convolutional=False):
        super().__init__()
        self.latent = latent
        self.convolutional = convolutional
        if convolutional:
            self.generator = nn.Sequential(
                nn.ConvTranspose2d(latent, 128, 7, bias=False), nn.BatchNorm2d(128), nn.ReLU(),
                nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
                nn.ConvTranspose2d(64, 1, 4, 2, 1), nn.Tanh())
            self.discriminator = nn.Sequential(
                nn.Conv2d(1, 64, 4, 2, 1), nn.LeakyReLU(0.2),
                nn.Conv2d(64, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
                nn.Conv2d(128, 1, 7), nn.Flatten())
            self.apply(self._initialize)
        else:
            self.generator = nn.Sequential(
                nn.Linear(latent, 256), nn.LeakyReLU(0.2),
                nn.Linear(256, 512), nn.LeakyReLU(0.2),
                nn.Linear(512, 784), nn.Tanh())
            self.discriminator = nn.Sequential(
                nn.Flatten(), nn.Linear(784, 512), nn.LeakyReLU(0.2),
                nn.Linear(512, 256), nn.LeakyReLU(0.2), nn.Linear(256, 1))

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(module.weight, 0, 0.02)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.normal_(module.weight, 1, 0.02)
            nn.init.zeros_(module.bias)

    def generate(self, z):
        if self.convolutional:
            z = z[:, :, None, None]
        return self.generator(z).reshape(-1, 1, 28, 28)

    def sample(self, z):
        return (self.generate(z) + 1) / 2

class TimeEmbedding(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t):
        freq = torch.exp(-math.log(10000) * torch.arange(
            self.dim // 2, device=t.device) / (self.dim // 2 - 1))
        phase = t.float()[:, None] * freq[None]
        return self.proj(torch.cat([phase.sin(), phase.cos()], dim=1))

class ResBlock(nn.Module):
    def __init__(self, cin, cout, time_dim=128):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.time = nn.Linear(time_dim, cout)
        self.norm2 = nn.GroupNorm(8, cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, t):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time(F.silu(t))[:, :, None, None]
        return self.skip(x) + self.conv2(F.silu(self.norm2(h)))

class NoiseUNet(nn.Module):
    def __init__(self, channels=32):
        super().__init__()
        c = channels
        self.time = TimeEmbedding()
        self.input = nn.Conv2d(1, c, 3, padding=1)
        self.down1 = ResBlock(c, c)
        self.down2 = ResBlock(c, c * 2)
        self.middle1 = ResBlock(c * 2, c * 4)
        self.middle2 = ResBlock(c * 4, c * 2)
        self.up2 = ResBlock(c * 4, c)
        self.up1 = ResBlock(c * 2, c)
        self.output = nn.Sequential(nn.GroupNorm(8, c), nn.SiLU(), nn.Conv2d(c, 1, 3, padding=1))

    def forward(self, x, timestep):
        t = self.time(timestep)
        h1 = self.down1(self.input(x), t)
        h2 = self.down2(F.avg_pool2d(h1, 2), t)
        h = self.middle2(self.middle1(F.avg_pool2d(h2, 2), t), t)
        h = self.up2(torch.cat([F.interpolate(h, size=h2.shape[-2:], mode='nearest'), h2], 1), t)
        h = self.up1(torch.cat([F.interpolate(h, size=h1.shape[-2:], mode='nearest'), h1], 1), t)
        return self.output(h)

class DDPM(nn.Module):
    def __init__(self, steps=200, channels=32):
        super().__init__()
        self.steps = steps
        self.network = NoiseUNet(channels)
        # Cosine schedule.
        t = torch.linspace(0, steps, steps + 1, dtype=torch.float64) / steps
        cumulative = torch.cos((t + 0.008) / 1.008 * math.pi / 2).square()
        cumulative = cumulative / cumulative[0]
        # Float64 menjaga presisi koefisien pada t=0.
        betas = (1 - cumulative[1:] / cumulative[:-1]).clamp(max=0.999)
        alphas = 1 - betas
        abar = torch.cumprod(alphas, 0)
        previous = F.pad(abar[:-1], (1, 0), value=1)
        for key, value in {
            'betas': betas, 'alphas': alphas, 'abar': abar,
            'posterior_var': betas * (1 - previous) / (1 - abar),
            'posterior_c1': betas * previous.sqrt() / (1 - abar),
            'posterior_c2': (1 - previous) * alphas.sqrt() / (1 - abar)
        }.items():
            self.register_buffer(key, value.float())

    def loss(self, x, t=None, noise=None):
        if t is None:
            t = torch.randint(self.steps, (len(x),), device=x.device)
        if noise is None:
            noise = torch.randn_like(x)
        a = self.abar[t, None, None, None]
        noisy = a.sqrt() * (2 * x - 1) + (1 - a).sqrt() * noise
        return F.mse_loss(self.network(noisy, t), noise)

    @torch.no_grad()
    def sample(self, n, trajectory=False, generator=None):
        device = self.betas.device
        x = torch.randn(n, 1, 28, 28, device=device, generator=generator)
        frames = [(x.clone() + 1) / 2] if trajectory else []
        marks = set(torch.linspace(self.steps - 1, 0, 7).long().tolist())
        for step in range(self.steps - 1, -1, -1):
            t = torch.full((n,), step, device=device, dtype=torch.long)
            epsilon = self.network(x, t)
            x0 = ((x - (1 - self.abar[step]).sqrt() * epsilon) /
                  self.abar[step].sqrt()).clamp(-1, 1)
            mean = self.posterior_c1[step] * x0 + self.posterior_c2[step] * x
            if step > 0:
                noise = torch.randn(x.shape, device=device, generator=generator)
                x = mean + self.posterior_var[step].sqrt() * noise
            else:
                x = mean
            if trajectory and step in marks:
                frames.append(((x.clone() + 1) / 2).clamp(0, 1))
        result = ((x + 1) / 2).clamp(0, 1)
        return (result, frames) if trajectory else result

def build_models(cfg):
    d = cfg['latent_dim']
    return {
        'classifier': Classifier(d), 'ae': Autoencoder(d),
        'sparse_ae': Autoencoder(d), 'vae': VAE(d),
        'frozen_decoder': Decoder(d),
        'gan': GAN(cfg['gan_latent_dim']),
        'dcgan': GAN(cfg['gan_latent_dim'], convolutional=True),
        'ddpm': DDPM(cfg['ddpm_steps'], cfg['ddpm_channels'])
    }
