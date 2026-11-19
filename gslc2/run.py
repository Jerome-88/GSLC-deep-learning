import argparse
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description='Eksperimen Fashion-MNIST Sesi 07')
    parser.add_argument('--epochs', type=int, help='Jumlah epoch untuk semua model; default 88')
    parser.add_argument('--output', type=Path, default=root / 'results_run',
                        help='Folder hasil baru; default results_run')
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    args = parser.parse_args()
    if args.epochs is not None and args.epochs < 1:
        parser.error('--epochs minimal 1')
    out = args.output.resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        parser.error('Folder output sudah berisi data. Pilih --output baru agar hasil lama tetap aman.')

    import torch

    from src.data import load_data
    from src.evaluate import audit, evaluate
    from src.models import build_models
    from src.train import train_model
    from src.utils import environment, set_seed, write_json

    cfg = json.loads((root / 'config.json').read_text(encoding='utf-8'))
    if args.epochs is not None:
        cfg['epochs'] = {name: args.epochs for name in cfg['epochs']}
    cfg['device'] = args.device
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA tidak tersedia pada environment ini.')
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available()
                          else ('cpu' if args.device == 'auto' else args.device))
    torch.set_num_threads(4)
    set_seed(cfg['seed'])
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'config.json', cfg)
    print('Device:', torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU')
    data = load_data(root / 'data', out, cfg['seed'], device, cfg['smoke'])
    models = build_models(cfg)
    models['sparse_ae'].load_state_dict(models['ae'].state_dict())
    start_weights = {
        name: {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        for name, model in models.items()
    }
    for model in models.values():
        model.to(device)
    summary = {
        name: {'size': len(ds),
               'class_counts': torch.bincount(ds.tensors[1], minlength=10).tolist()}
        for name, ds in data.items()
    }
    write_json(out / 'data_summary.json', summary)
    write_json(out / 'parameter_counts.json', {
        name: sum(p.numel() for p in model.parameters()) for name, model in models.items()
    })
    write_json(out / 'environment.json', environment(device))
    for name in cfg['epochs']:
        train_model(name, models, data, cfg, out, device, start_weights)
    evaluate(models, data, cfg, out, device)
    checks = audit(out)
    stats = json.loads((out / 'classifier_diagnostics.json').read_text(encoding='utf-8'))
    print(f'Test accuracy classifier: {stats["test_accuracy"]:.2%}')
    print(f'{checks["checks_passed"]} pemeriksaan lulus, {len(checks["failures"])} gagal.')
    for message in checks['failures']:
        print(message)
    print('Hasil:', out)
    if checks['failures']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
