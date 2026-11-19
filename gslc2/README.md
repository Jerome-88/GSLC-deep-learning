# Fashion-MNIST: AE, VAE, GAN, dan Diffusion

Tugas Deep Learning Sesi 07. Jerome Maxcellino B, NIM 2802412894, kelas LB01.

Eksperimen membandingkan Vanilla AE, Sparse AE, VAE, Vanilla GAN, DCGAN, dan DDPM. Ada juga classifier dan frozen decoder untuk menguji rekonstruksi dari fitur classifier.

## File utama

| File | Isi |
|---|---|
| `experiment.ipynb` | Seluruh kode dan output eksperimen dalam satu notebook |
| `src/models.py` | Enam model, classifier, dan decoder |
| `src/data.py` | Download dataset dan pembagian train/validation/test |
| `src/train.py` | Loss, validation, dan training per model |
| `src/evaluate.py` | Metrik, percobaan classifier, dan pemeriksaan hasil |
| `src/figures.py` | Lima gambar hasil yang diminta tugas |
| `src/utils.py` | Seed, penyimpanan JSON, dan informasi environment |
| `run.py` | Menjalankan eksperimen dari terminal |
| `config.json` | Pengaturan run 88 epoch |
| `requirements.txt` | Versi library pada environment eksperimen |
| `results/` | Metrik, catatan epoch, indeks split, dan grafik run yang sudah selesai |

## Kenapa ada kode modular?

Ketentuan tugas pada Bagian I meminta repositori memuat skrip modular PyTorch. Karena itu, fungsi dari notebook dipisahkan ke folder `src/` berdasarkan kegunaannya: data, model, training, evaluasi, plot, dan fungsi pendukung.

Versi modular memakai model dan pengaturan yang sama dengan notebook. `run.py` menghubungkan modul-modul tersebut agar eksperimen bisa dijalankan dari terminal. Notebook tetap lengkap dan bisa dijalankan sendiri tanpa mengimpor `src/`. Hasil yang sudah tersimpan bisa dilihat tanpa menjalankan ulang salah satu versi.

## Instalasi

Gunakan Python 3.11. Eksperimen tersimpan dijalankan dengan Python 3.11.18, PyTorch 2.11.0+cu128, dan RTX 5050 Laptop GPU.

Dari root repositori:

```powershell
cd gslc2
```

Jika membuat environment baru untuk GPU NVIDIA, pasang PyTorch CUDA terlebih dahulu:

```powershell
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

Jika environment sudah memiliki PyTorch CUDA yang sesuai, cukup jalankan perintah instalasi requirements. Untuk CPU, gunakan `--index-url https://download.pytorch.org/whl/cpu` pada perintah pertama. Training dan sampling DDPM akan lebih lambat pada CPU.

Perintah instalasi CUDA mengacu pada [daftar versi resmi PyTorch](https://pytorch.org/get-started/previous-versions/).

## Menjalankan notebook

Buka `experiment.ipynb` di VS Code dan pilih kernel dari environment yang sudah dipasangi requirements. Output run yang selesai sudah tersimpan di notebook, jadi hasilnya bisa dilihat tanpa menjalankan ulang sel.

Untuk eksperimen baru, pilih Restart Kernel lalu Run All. Semua model dilatih dari bobot awal dan setiap epoch dicetak langsung. Run All menulis ulang hasil di `results/`, jadi simpan salinan hasil lama terlebih dahulu jika masih diperlukan. Biarkan `tes_cepat = False` untuk eksperimen lengkap.

Fashion-MNIST diunduh otomatis saat pertama dijalankan. Bobot Inception untuk FID juga diunduh saat pertama dipakai, sehingga kedua tahap ini membutuhkan internet.

## Menjalankan skrip modular

```powershell
python run.py
```

Default-nya memakai 88 epoch dan menyimpan hasil baru di `results_run/`. Skrip menolak folder output yang sudah berisi data agar hasil sebelumnya tidak tertimpa.

Pilihan pengaturan:

```powershell
python run.py --epochs 88 --device cuda --output results_run_baru
python run.py --help
```

Menjalankan `run.py` benar-benar melatih ulang semua model. Perintah ini tidak diperlukan untuk melihat hasil yang sudah ada di `results/`.

## Pengaturan dan evaluasi

- Dataset: 70.000 gambar Fashion-MNIST, dibagi secara stratified menjadi 56.000 train, 7.000 validation, dan 7.000 test.
- Seed 42, batch size 128, tanpa augmentasi, dan 88 epoch untuk semua model.
- Laten AE/Sparse AE/VAE/classifier: 64; noise GAN/DCGAN: 100; DDPM: 200 langkah.
- Classifier dipilih dari akurasi validation tertinggi. AE, Sparse AE, VAE, frozen decoder, dan DDPM dipilih dari loss validation terendah. GAN/DCGAN memakai epoch terakhir.
- Rekonstruksi dinilai dengan MSE, PSNR per gambar, dan SSIM. FID memakai 7.000 gambar asli serta 7.000 gambar buatan per generator dengan Inception-2048.
- Latency memakai batch 64, 3 warmup, 10 pengulangan, dan sinkronisasi CUDA. AE diukur encode+decode, sedangkan VAE/GAN/DCGAN/DDPM diukur saat membuat gambar.
- Training DDPM memakai bfloat16 pada GPU yang mendukungnya. Evaluasi memakai float32.

## Hasil run yang disertakan

| Model | Test MSE | Test SSIM | FID |
|---|---:|---:|---:|
| Vanilla AE | 0,007386 | 0,8508 | N/A |
| Sparse AE | 0,007482 | 0,8478 | N/A |
| VAE | 0,012512 | 0,7764 | 70,57 |
| Vanilla GAN | N/A | N/A | 119,65 |
| DCGAN | N/A | N/A | 17,25 |
| DDPM | N/A | N/A | 14,39 |

Akurasi test classifier: 89,83%. Selengkapnya ada di [metrics.csv](results/metrics.csv), [catatan evaluasi](results/evaluation_protocol.json), [pemeriksaan hasil](results/audit.json), dan [folder grafik](results/figures/).

Ini hasil satu seed pada gambar 28 x 28. Histogram prediksi classifier belum membuktikan seluruh variasi gambar tercakup. Pengujian mobile, data cacat, dan gambar resolusi tinggi belum dilakukan.

Laporan: [Word](TUGAS_DL_SESI07_2802412894_JEROME_MAXCELLINO_B.docx) dan [PDF](TUGAS_DL_SESI07_2802412894_JEROME_MAXCELLINO_B.pdf). Ekspor ulang PDF dari Word jika naskah Word masih disunting.

## Sumber kode dan teori

Notebook disusun untuk tugas ini dengan bantuan AI. Rujukan berikut dipakai untuk memeriksa implementasi dan rumus; kode ini bukan salinan utuh dari satu repositori.

- [Fashion-MNIST, Zalando](https://github.com/zalandoresearch/fashion-mnist)
- [Training loop PyTorch](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)
- [Contoh VAE PyTorch](https://github.com/pytorch/examples/blob/main/vae/main.py)
- [Tutorial DCGAN PyTorch](https://github.com/pytorch/tutorials/blob/main/beginner_source/dcgan_faces_tutorial.py)
- [Implementasi diffusion OpenAI](https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py)
- [FID TorchMetrics](https://lightning.ai/docs/torchmetrics/stable/image/frechet_inception_distance)

Layer MLP, ukuran jaringan untuk 28 x 28, U-Net kecil, dan percobaan classifier disesuaikan untuk tugas ini. Referensi teori lengkap ada di laporan.

