import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid

from .utils import set_seed, write_json

NAMES = {'ae': 'Vanilla AE', 'sparse_ae': 'Sparse AE', 'vae': 'VAE',
         'gan': 'Vanilla GAN', 'dcgan': 'DCGAN', 'ddpm': 'DDPM'}
CLASSES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def image(ax, img, title=None):
    ax.imshow(img.detach().cpu().squeeze().numpy(), cmap='gray', vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9)

def grid(ax, img, nrow=4, title=None):
    pixels = make_grid(img.detach().cpu().float(), nrow=nrow, padding=2, pad_value=0.15)
    ax.imshow(pixels.permute(1, 2, 0).clamp(0, 1).numpy())
    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=11)

def save(fig, out, stem):
    fig.savefig(out / 'figures' / f'{stem}.png', dpi=240, bbox_inches='tight')
    fig.savefig(out / 'figures' / f'{stem}.pdf', bbox_inches='tight')
    plt.close(fig)

@torch.inference_mode()
def make_figures(models, data, cfg, out, device, fake_images):
    set_seed(cfg['seed'] + 3000)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    x, y = data['test'].tensors
    # Gambar pertama setiap kelas pada test set.
    indices = [int(torch.where(y == label)[0][0]) for label in range(10)]
    selected = x[indices]
    info = {'reconstruction_test_positions': indices,
                  'reconstruction_selection': 'first occurrence of each class in persisted test order'}
    recons = [selected, models['ae'](selected), models['sparse_ae'](selected),
              models['vae'].decoder(models['vae'].encode(selected)[0])]
    fig, axes = plt.subplots(4, 10, figsize=(15, 6))
    for r, (row, title) in enumerate(zip(recons, ['Asli', 'Vanilla AE', 'Sparse AE', 'VAE'])):
        for c, sample in enumerate(row):
            image(axes[r, c], sample, CLASSES[c] if r == 0 else None)
        axes[r, 0].set_ylabel(title, fontsize=10)
    fig.suptitle('Gambar 1. Rekonstruksi')
    fig.tight_layout()
    save(fig, out, 'fig1_reconstructions')

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.5))
    for ax, name in zip(axes, ['vae', 'gan', 'dcgan', 'ddpm']):
        grid(ax, fake_images[name][:16].float() / 255, title=NAMES[name])
    fig.suptitle('Gambar 2. Sampel Acak')
    fig.tight_layout()
    save(fig, out, 'fig2_random_sampling')

    clf = models['classifier']
    latent = clf.encode(selected)
    frozen = models['frozen_decoder'](latent)
    fig = plt.figure(figsize=(16, 9), layout='constrained')
    outer = fig.subfigures(2, 2)
    panels = [
        ('A. Asli test images', [('Asli', selected)]),
        ('B. Transpose Bobot', [('Dari Label Asli', clf.transpose_class(y[indices])),
                                            ('Dari Fitur', clf.transpose_latent(latent))]),
        ('C. Rekonstruksi', [('Classifier Beku + Decoder', frozen), ('Vanilla AE', recons[1])]),
        ('D. Sampling Fitur Classifier', [('Gaussian + Decoder',
          torch.load(out / 'samples/classifier_prior.pt', weights_only=True)[:10])])]
    for subfig, (title, rows) in zip(outer.flat, panels):
        subfig.suptitle(title, fontsize=11)
        axes = subfig.subplots(len(rows), 1, squeeze=False)
        for ax, (label, samples) in zip(axes[:, 0], rows):
            grid(ax, samples, nrow=5, title=label)
    fig.suptitle('Gambar 3. Eksperimen Classifier', fontsize=14)
    save(fig, out, 'fig3_classifier_experiments')

    # Kelas endpoint GAN ditentukan oleh classifier.
    endpoint_classes = [0, 7]
    endpoint_positions = [int(torch.where(y == label)[0][0]) for label in endpoint_classes]
    endpoints = x[endpoint_positions]
    alpha = torch.tensor([0., .2, .4, .6, .8, 1.], device=device)[:, None]
    fig, axes = plt.subplots(5, 6, figsize=(11, 9))
    info['interpolation_real_test_positions'] = endpoint_positions
    info['gan_endpoint_selection'] = 'first generated endpoint predicted as target class by trained classifier; no labels condition generator'
    for r, name in enumerate(['ae', 'sparse_ae', 'vae', 'gan', 'dcgan']):
        model = models[name]
        if name in ('ae', 'sparse_ae'):
            z = model.encoder(endpoints)
            result = model.decoder((1 - alpha) * z[0] + alpha * z[1])
        elif name == 'vae':
            z = model.encode(endpoints)[0]
            result = model.decoder((1 - alpha) * z[0] + alpha * z[1])
        else:
            pool = torch.randn(1024, model.latent, device=device)
            images = model.sample(pool)
            scores = clf(images).softmax(1)
            chosen = []
            for label in endpoint_classes:
                matches = torch.where(scores.argmax(1) == label)[0]
                chosen.append(int(matches[0]) if len(matches) else int(scores[:, label].argmax()))
            z = pool[chosen]
            info[f'{name}_endpoint_predicted_classes'] = scores[chosen].argmax(1).cpu().tolist()
            info[f'{name}_endpoint_confidences'] = scores[chosen].max(1).values.cpu().tolist()
            result = model.sample((1 - alpha) * z[0] + alpha * z[1])
        for c, sample in enumerate(result):
            image(axes[r, c], sample, f'alpha={float(alpha[c, 0]):.1f}' if r == 0 else None)
        axes[r, 0].set_ylabel(NAMES[name])
    fig.suptitle('Gambar 4. Interpolasi Laten', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .94))
    save(fig, out, 'fig4_latent_interpolation')

    fig = plt.figure(figsize=(16, 12), layout='constrained')
    spec = fig.add_gridspec(3, 3, height_ratios=[1, 1, .8])
    for index, name in enumerate(['ae', 'sparse_ae', 'vae', 'gan', 'dcgan', 'ddpm']):
        ax = fig.add_subplot(spec[index // 3, index % 3])
        history = json.loads((out / f'{name}_history.json').read_text())
        epochs = [row['epoch'] for row in history]
        ax.plot(epochs, [row['train_loss'] for row in history], label='G' if name in ('gan', 'dcgan') else 'Train', color='black')
        if name in ('gan', 'dcgan'):
            ax.plot(epochs, [row['discriminator_loss'] for row in history], label='D', color='black', linestyle='--')
        else:
            ax.plot(epochs, [row['val_loss'] for row in history], label='Val', color='black', linestyle='--')
        ax.set(title=NAMES[name], xlabel='Epoch', ylabel='Loss')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    _, frames = models['ddpm'].sample(1, trajectory=True)
    trajectory_spec = spec[2, :].subgridspec(1, len(frames))
    marks = sorted(set(torch.linspace(cfg['ddpm_steps'] - 1, 0, 7).long().tolist()), reverse=True)
    labels = [f'x_T (T={cfg["ddpm_steps"]})'] + [f'x_{t}' for t in marks]
    for i, (snapshot, label) in enumerate(zip(frames, labels)):
        image(fig.add_subplot(trajectory_spec[0, i]), snapshot.clamp(0, 1), label)
    fig.suptitle('Gambar 5. Loss Training dan Denoising', fontsize=13)
    save(fig, out, 'fig5_training_and_diffusion')
    write_json(out / 'figure_provenance.json', info)
