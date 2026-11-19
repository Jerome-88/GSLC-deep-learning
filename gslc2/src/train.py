import contextlib
import time

import torch
from torch.nn import functional as F

from .data import loader
from .utils import set_seed, sync_gpu, write_json


def get_loss(name, model, x, y, clf, cfg):
    parts = {}
    logits = None
    if name == 'classifier':
        logits = model(x)
        loss = F.cross_entropy(logits, y)
    elif name == 'frozen_decoder':
        with torch.no_grad():
            z = clf.encode(x)
        loss = F.mse_loss(model(z), x)
        parts['reconstruction_loss'] = loss
    elif name in ('ae', 'sparse_ae'):
        z = model.encoder(x)
        recon = F.mse_loss(model.decoder(z), x)
        loss = recon
        parts['reconstruction_loss'] = recon
        if name == 'sparse_ae':
            penalty = cfg['sparsity_lambda'] * z.abs().mean()
            loss = loss + penalty
            parts['sparsity_penalty'] = penalty
    elif name == 'vae':
        recon, mu, logvar = model(x)
        bce = F.binary_cross_entropy(recon, x, reduction='sum') / len(x)
        kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(1).mean()
        loss = bce + kl
        parts = {'reconstruction_loss': bce, 'kl_loss': kl}
    elif name == 'ddpm':
        loss = model.loss(x)
    else:
        raise ValueError(name)
    return loss, parts, logits

@torch.no_grad()
def validate(name, model, dataset, clf, cfg, device):
    model.eval()
    total, count, correct = 0., 0, 0
    # Noise validasi tetap setiap epoch.
    with torch.random.fork_rng(devices=[torch.cuda.current_device()] if device.type == 'cuda' else []):
        torch.manual_seed(cfg['seed'] + 991)
        for x, y in loader(dataset, cfg['batch_size']):
            loss, _, logits = get_loss(name, model, x, y, clf, cfg)
            if logits is not None:
                correct += int((logits.argmax(1) == y).sum())
            total += float(loss) * len(x)
            count += len(x)
    return total / count, correct / count if name == 'classifier' else None

def train_model(name, models, data, cfg, out, device, start_weights):
    set_seed(cfg['seed'])
    model = models[name].to(device)
    model.load_state_dict(start_weights[name])
    model.requires_grad_(True)
    clf = models['classifier'].to(device)
    if name == 'frozen_decoder':
        clf.eval().requires_grad_(False)
    model_dir = out / 'checkpoints'
    model_dir.mkdir(exist_ok=True)
    best_path = model_dir / f'{name}_best.pt'
    is_gan = name in ('gan', 'dcgan')
    lr = cfg['gan_learning_rate'] if is_gan else (
        cfg['ddpm_learning_rate'] if name == 'ddpm' else cfg['learning_rate'])
    if is_gan:
        opts = [torch.optim.Adam(model.generator.parameters(), lr=lr, betas=(0.5, 0.999)),
                      torch.optim.Adam(model.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))]
    else:
        opts = [torch.optim.Adam(model.parameters(), lr=lr)]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opts[0], factor=0.5, patience=5, min_lr=1e-5) if name == 'classifier' else None
    history, best = [], float('inf')
    epochs = cfg['epochs'][name]
    print(f'{name}: training {epochs} epoch', flush=True)
    for epoch in range(epochs):
        model.train()
        total, total_d, count, correct = 0., 0., 0, 0
        components = {}
        sync_gpu(device)
        begin = time.perf_counter()
        for x, y in loader(data['train'], cfg['batch_size'], train=True,
                           seed=cfg['seed'] + epoch):
            if is_gan:
                opt_g, opt_d = opts
                model.discriminator.requires_grad_(True)
                opt_d.zero_grad(set_to_none=True)
                real_logits = model.discriminator(2 * x - 1)
                z = torch.randn(len(x), model.latent, device=device)
                fake = model.generate(z)
                fake_logits = model.discriminator(fake.detach())
                d_loss = (F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits)) +
                          F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits)))
                d_loss.backward()
                opt_d.step()
                model.discriminator.requires_grad_(False)
                opt_g.zero_grad(set_to_none=True)
                generated_logits = model.discriminator(fake)
                loss = F.binary_cross_entropy_with_logits(generated_logits, torch.ones_like(generated_logits))
                loss.backward()
                opt_g.step()
                total_d += float(d_loss.detach()) * len(x)
            else:
                opts[0].zero_grad(set_to_none=True)
                use_amp = (name == 'ddpm' and device.type == 'cuda' and torch.cuda.is_bf16_supported())
                amp = torch.autocast('cuda', dtype=torch.bfloat16) if use_amp else contextlib.nullcontext()
                with amp:
                    loss, parts, logits = get_loss(name, model, x, y, clf, cfg)
                    if logits is not None:
                        correct += int((logits.detach().argmax(1) == y).sum())
                for key, value in parts.items():
                    components[key] = components.get(key, 0.) + float(value.detach()) * len(x)
                loss.backward()
                if name == 'ddpm':
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opts[0].step()
            if not torch.isfinite(loss):
                raise FloatingPointError(f'{name}: nonfinite loss at epoch {epoch + 1}')
            total += float(loss.detach()) * len(x)
            count += len(x)
        sync_gpu(device)
        seconds = time.perf_counter() - begin
        row = {'epoch': epoch + 1, 'train_loss': total / count,
                  'training_seconds': seconds, 'learning_rate': opts[0].param_groups[0]['lr']}
        row.update({key: value / count for key, value in components.items()})
        if name == 'classifier':
            row['train_accuracy'] = correct / count
        if is_gan:
            row['discriminator_loss'] = total_d / count
            score = -(epoch + 1)  # GAN memakai epoch terakhir.
        else:
            val_loss, val_accuracy = validate(name, model, data['val'], clf, cfg, device)
            row['val_loss'] = val_loss
            if val_accuracy is not None:
                row['val_accuracy'] = val_accuracy
            score = -val_accuracy if name == 'classifier' else val_loss
            if scheduler is not None:
                scheduler.step(val_loss)
        history.append(row)
        if score < best:
            best = score
            torch.save({'model': model.state_dict(), 'epoch': epoch + 1, 'selection_score': score}, best_path)
        write_json(out / f'{name}_history.json', history)
        write_json(out / 'progress.json', {'stage': 'training', 'model': name,
                   'epoch': epoch + 1, 'target_epochs': epochs, 'last_record': row})
        line = f'epoch {epoch + 1}/{epochs} {"g_loss" if is_gan else "loss"}={row["train_loss"]:.6f}'
        if 'discriminator_loss' in row:
            line += f' d_loss={row["discriminator_loss"]:.6f}'
        if 'val_loss' in row:
            line += f' val_loss={row["val_loss"]:.6f}'
        if name == 'classifier':
            line += f' train_acc={row["train_accuracy"]:.2%} val_acc={row["val_accuracy"]:.2%}'
        print(line + f' time={seconds:.1f}s', flush=True)
    selected = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(selected['model'])
    model.eval()
    write_json(out / f'{name}_selection.json', {'epoch': selected['epoch'],
               'criterion': 'last epoch' if is_gan else ('validation accuracy' if name == 'classifier' else 'validation objective'),
               'epochs_trained': len(history)})
    return history
