import csv
import json
import math
import time

import numpy as np
import torch
from torchmetrics.functional.image import structural_similarity_index_measure
from torchmetrics.image.fid import FrechetInceptionDistance

from .data import loader
from .figures import make_figures
from .utils import set_seed, sync_gpu, write_json


@torch.inference_mode()
def reconstruct(name, model, x, clf):
    if name == 'vae':
        return model.decoder(model.encode(x)[0])  # Rekonstruksi dari posterior mean.
    if name == 'frozen_decoder':
        return model(clf.encode(x))
    if name == 'transpose_class':
        return clf.transpose_class(clf(x).argmax(1))
    if name == 'transpose_latent':
        return clf.transpose_latent(clf.encode(x))
    return model(x)

@torch.inference_mode()
def reconstruction_metrics(name, model, dataset, clf, batch_size):
    values = []
    for x, _ in loader(dataset, batch_size):
        result = reconstruct(name, model, x, clf).clamp(0, 1)
        mse = (result - x).square().flatten(1).mean(1)
        psnr = -10 * mse.clamp_min(1e-12).log10()
        ssim = structural_similarity_index_measure(result, x, data_range=1., reduction='none')
        values.append(torch.stack([mse, psnr, ssim], 1).cpu())
    stacked = torch.cat(values).double()
    mean = stacked.mean(0)
    return dict(zip(['test_mse', 'test_psnr_db', 'test_ssim'], mean.tolist()))

@torch.inference_mode()
def sample_model(name, model, n, device):
    if name == 'ddpm':
        return model.sample(n)
    if name == 'vae':
        return model.decoder(torch.randn(n, model.mu.out_features, device=device))
    return model.sample(torch.randn(n, model.latent, device=device))

@torch.inference_mode()
def measure_time(fn, cfg, device):
    for _ in range(cfg['latency_warmup']):
        fn()
    timings = []
    for _ in range(cfg['latency_repeats']):
        sync_gpu(device)
        start = time.perf_counter()
        fn()
        sync_gpu(device)
        timings.append((time.perf_counter() - start) * 1000)
    return {'latency_mean_ms_batch64': float(np.mean(timings)),
            'latency_median_ms_batch64': float(np.median(timings)),
            'latency_std_ms_batch64': float(np.std(timings)),
            'latency_raw_ms': timings}

@torch.inference_mode()
def generate_samples(name, model, cfg, out, device):
    path = out / 'samples' / f'{name}.pt'
    set_seed(cfg['seed'] + 1000)
    batches, count = [], 0
    while count < cfg['fid_samples']:
        n = min(cfg['sampling_batch_size'], cfg['fid_samples'] - count)
        images = sample_model(name, model, n, device)
        batches.append((images.clamp(0, 1) * 255).round().to(torch.uint8).cpu())
        count += n
        if count % (cfg['sampling_batch_size'] * 10) == 0 or count == cfg['fid_samples']:
            print(f'{name}: {count}/{cfg["fid_samples"]} sampel', flush=True)
        write_json(out / 'progress.json', {'stage': 'sampling', 'model': name,
                   'samples': count, 'target': cfg['fid_samples']})
    result = torch.cat(batches)
    torch.save(result, path)
    return result

@torch.inference_mode()
def check_classifier(models, data, cfg, out):
    clf = models['classifier']
    x, y = data['test'].tensors
    features, predictions = [], []
    for xb, _ in loader(data['test'], cfg['batch_size']):
        features.append(clf.encode(xb))
        predictions.append(clf(xb).argmax(1))
    z = torch.cat(features)
    accuracy = float((torch.cat(predictions) == y).float().mean())
    result = {'test_accuracy': accuracy, 'minimum_required_accuracy': 0.88,
              'accuracy_requirement_met': accuracy >= 0.88,
              'sampling_statistics_source': 'test split, explicitly required by assignment; not used for training',
              'feature_mean': z.mean(0).cpu().tolist(),
              'feature_std': z.std(0, unbiased=False).cpu().tolist()}
    for label, model in [('classifier', clf), ('ae', models['ae']), ('sparse_ae', models['sparse_ae'])]:
        encoded = []
        for xb, _ in loader(data['test'], cfg['batch_size']):
            encoded.append(model.encode(xb) if label == 'classifier' else model.encoder(xb))
        latent = torch.cat(encoded)
        result[f'{label}_latent_zero_fraction'] = float((latent.abs() < 1e-6).float().mean())
        result[f'{label}_latent_mean_abs'] = float(latent.abs().mean())
    weights = {}
    for name in ['fc1', 'fc2', 'fc3']:
        w = getattr(clf, name).weight.detach()
        identity = torch.eye(w.shape[1], device=w.device)
        weights[name] = {'shape_out_in': list(w.shape),
                         'rank': int(torch.linalg.matrix_rank(w)),
                         'relative_gram_identity_error': float(torch.linalg.norm(w.T @ w - identity) /
                                                               torch.linalg.norm(identity))}
    result['weights'] = weights
    result['transpose_latent_metrics'] = reconstruction_metrics('transpose_latent', None,
                                                data['test'], clf, cfg['batch_size'])
    # Metrik proyeksi kelas memakai prediksi classifier.
    result['transpose_predicted_class_metrics'] = reconstruction_metrics('transpose_class', None,
                                                data['test'], clf, cfg['batch_size'])
    set_seed(cfg['seed'] + 2000)
    samples = z.mean(0) + z.std(0, unbiased=False) * torch.randn(64, z.shape[1], device=z.device)
    result['sampled_latent_negative_fraction'] = float((samples < 0).float().mean())
    before_relu = clf.fc1(x.flatten(1))
    result['classifier_first_relu_negative_input_fraction'] = float((before_relu < 0).float().mean())
    torch.save(models['frozen_decoder'](samples).cpu(), out / 'samples/classifier_prior.pt')
    write_json(out / 'classifier_diagnostics.json', result)
    return result

@torch.inference_mode()
def evaluate(models, data, cfg, out, device):
    (out / 'samples').mkdir(exist_ok=True)
    (out / 'figures').mkdir(exist_ok=True)
    set_seed(cfg['seed'] + 500)
    rows = []
    x64 = data['test'].tensors[0][:64]
    for name in ['ae', 'sparse_ae', 'vae', 'gan', 'dcgan', 'ddpm', 'frozen_decoder']:
        print(f'Evaluasi {name}', flush=True)
        model = models[name]
        history = json.loads((out / f'{name}_history.json').read_text(encoding='utf-8'))
        row = {'model': name,
               'latent_dimension': cfg['gan_latent_dim'] if name in ('gan', 'dcgan') else (
                   '1x28x28 image noise' if name == 'ddpm' else cfg['latent_dim']),
               'epochs_trained': len(history),
               'training_mean_seconds_epoch': float(np.mean([r['training_seconds'] for r in history])),
               'test_mse': None, 'test_psnr_db': None, 'test_ssim': None, 'fid_inception2048': None,
               'checkpoint_selection': json.loads((out / f'{name}_selection.json').read_text())['criterion'],
               'stability_note': 'Finite recorded losses; inspect curves and samples for oscillation/collapse.'}
        last_losses = [r['train_loss'] for r in history[-10:]]
        row['last10_loss_coefficient_variation'] = float(np.std(last_losses) / max(abs(np.mean(last_losses)), 1e-12))
        if name in ('ae', 'sparse_ae', 'vae', 'frozen_decoder'):
            row.update(reconstruction_metrics(name, model, data['test'], models['classifier'], cfg['batch_size']))
        if name in ('ae', 'sparse_ae', 'frozen_decoder'):
            fn = lambda name=name, model=model: reconstruct(name, model, x64, models['classifier'])
            row['latency_operation'] = 'encode + decode' if name != 'frozen_decoder' else 'frozen classifier encode + decode'
        else:
            fn = lambda name=name, model=model: sample_model(name, model, 64, device)
            row['latency_operation'] = f'{cfg["ddpm_steps"]}-step ancestral DDPM sampling' if name == 'ddpm' else 'prior noise + generator/decoder'
        row.update(measure_time(fn, cfg, device))
        rows.append(row)
        write_json(out / 'metrics_partial.json', rows)
    stats = check_classifier(models, data, cfg, out)
    fake_images = {}
    for name in ['vae', 'gan', 'dcgan', 'ddpm']:
        fake_images[name] = generate_samples(name, models[name], cfg, out, device)
    generation_diagnostics = {'limitation': 'Classifier-assigned labels are a proxy; class counts do not prove within-class mode coverage.'}
    for name, samples in fake_images.items():
        probs = []
        for batch in samples.split(cfg['batch_size']):
            probs.append(models['classifier'](batch.to(device).float() / 255).softmax(1).cpu())
        probs = torch.cat(probs)
        counts = torch.bincount(probs.argmax(1), minlength=10)
        dist = counts.float() / len(samples)
        generation_diagnostics[name] = {
            'classifier_predicted_class_counts': counts.tolist(),
            'class_distribution_entropy_nats': float(-(dist * dist.clamp_min(1e-12).log()).sum()),
            'max_possible_class_entropy_nats': float(np.log(10)),
            'mean_classifier_confidence': float(probs.max(1).values.mean())}
    write_json(out / 'generation_diagnostics.json', generation_diagnostics)
    print('Menghitung FID', flush=True)
    # Inception float32; akumulasi statistik bawaan TorchMetrics float64.
    fid = FrechetInceptionDistance(feature=2048, normalize=False,
                                   reset_real_features=False).to(device)
    real = (data['test'].tensors[0][:cfg['fid_samples']] * 255).round().to(torch.uint8)
    batch_size = cfg['fid_batch_size']
    for batch in real.split(batch_size):
        fid.update(batch.repeat(1, 3, 1, 1), real=True)
    for name, images in fake_images.items():
        fid.reset()
        for batch in images.split(batch_size):
            fid.update(batch.to(device).repeat(1, 3, 1, 1), real=False)
        value = float(fid.compute())
        next(row for row in rows if row['model'] == name)['fid_inception2048'] = value
        write_json(out / 'metrics_partial.json', rows)
        print(f'FID {name}: {value:.4f}', flush=True)
    del fid
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    write_json(out / 'metrics.json', rows)
    columns = [key for key in rows[0] if key != 'latency_raw_ms']
    with (out / 'metrics.csv').open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    write_json(out / 'evaluation_protocol.json', {
        'test_images': len(data['test']), 'fid_real_images': len(real),
        'fid_synthetic_images_per_model': cfg['fid_samples'],
        'fid_feature': 'torch-fidelity pretrained Inception-v3 pool3 2048, torchmetrics',
        'fid_preprocessing': 'round to uint8 [0,255], repeat grayscale into RGB; internal Inception resize',
        'fid_limitations': 'ImageNet features are not fashion-specific; finite-sample bias; same sample count for every model.',
        'reconstruction': 'mean per-image MSE; mean per-image PSNR (data_range=1); Gaussian SSIM kernel=11 sigma=1.5',
        'vae_reconstruction': 'decode posterior mean, not a random posterior draw',
        'latency': 'batch64, float32, device resident, synchronized wall clock; excludes disk/CPU data transfers',
        'latency_warmup': cfg['latency_warmup'], 'latency_repeats': cfg['latency_repeats'],
        'training_time': 'train loop only; excludes validation, checkpoint IO, download; includes initial warmup epoch',
        'checkpoint_selection': 'validation objective; classifier validation accuracy; GAN/DCGAN last epoch',
        'classifier_requirement_met': stats['accuracy_requirement_met'],
        'smoke_run': cfg['smoke'],
        'scope': 'One seed. No hardware-mobile or high-resolution deployment validation.'})
    make_figures(models, data, cfg, out, device, fake_images)
    write_json(out / 'progress.json', {'stage': 'complete',
               'classifier_accuracy_requirement_met': stats['accuracy_requirement_met'],
               'smoke_run': cfg['smoke']})
    print('Metrik dan gambar tersimpan.', flush=True)

def audit(out, allow_smoke=False):
    def read(name):
        return json.loads((out / name).read_text(encoding='utf-8'))

    checks, failures = [], []
    def check(ok, message):
        (checks if ok else failures).append(message)

    cfg = read('config.json')
    split = np.load(out / 'split_indices.npz')
    parts = [split[name] for name in ['train', 'val', 'test']]
    check([len(x) for x in parts] == [56000, 7000, 7000], 'Ukuran data 80/10/10')
    check(len(np.unique(np.concatenate(parts))) == 70000, 'Data tidak tumpang tindih')
    rows = {row['model']: row for row in read('metrics.json')}
    expected = {'ae', 'sparse_ae', 'vae', 'gan', 'dcgan', 'ddpm', 'frozen_decoder'}
    check(set(rows) == expected, 'Semua model sudah dievaluasi')
    for name, row in rows.items():
        keys = ['latency_mean_ms_batch64', 'training_mean_seconds_epoch']
        if name in ('ae', 'sparse_ae', 'vae', 'frozen_decoder'):
            keys += ['test_mse', 'test_psnr_db', 'test_ssim']
        if name in ('vae', 'gan', 'dcgan', 'ddpm'):
            keys += ['fid_inception2048']
        check(all(row[key] is not None and math.isfinite(row[key]) for key in keys),
              f'{name}: metrik valid')
    for name, epochs in cfg['epochs'].items():
        history = read(f'{name}_history.json')
        check([row['epoch'] for row in history] == list(range(1, epochs + 1)),
              f'{name}: semua epoch selesai')
        check(all(math.isfinite(row['train_loss']) for row in history), f'{name}: loss valid')
        check((out / f'checkpoints/{name}_best.pt').is_file(), f'{name}: model tersimpan')
    acc = read('classifier_diagnostics.json')['test_accuracy']
    check(acc >= .88 or (allow_smoke and cfg['smoke']), f'Akurasi classifier: {acc:.2%}')
    figures = ['fig1_reconstructions', 'fig2_random_sampling', 'fig3_classifier_experiments',
               'fig4_latent_interpolation', 'fig5_training_and_diffusion']
    for name in figures:
        check(all((out / 'figures' / f'{name}.{ext}').is_file() for ext in ['png', 'pdf']),
              f'{name}: gambar tersimpan')
    if not cfg['smoke']:
        protocol = read('evaluation_protocol.json')
        check(protocol['test_images'] == 7000 and protocol['fid_real_images'] == 7000 and
              protocol['fid_synthetic_images_per_model'] == 7000, 'Evaluasi memakai 7.000 gambar')
    check(read('progress.json')['stage'] == 'complete', 'Eksperimen selesai')
    result = {'passed': not failures, 'checks_passed': len(checks), 'failures': failures}
    write_json(out / 'audit.json', result)
    return result
