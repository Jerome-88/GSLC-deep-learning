# Studi Komparatif Paradigma Transfer Learning pada Klasifikasi Ras Hewan (Oxford-IIIT Pet)

Repositori ini memuat implementasi eksperimen, evaluasi empiris, dan analisis komparatif paradigma Transfer Learning pada Convolutional Neural Networks (CNN) menggunakan dataset Oxford-IIIT Pet (subset 10 kelas). Proyek ini disusun untuk memenuhi tugas Sesi 05 Mata Kuliah Deep Learning: *Advanced CNNs & Transfer Learning*.

---

## 1. Ringkasan Eksperimen

Eksperimen dirancang untuk menguji pengaruh inisialisasi bobot, strategi pembekuan layer (*layer freezing*), augmentasi citra, dan pemilihan arsitektur terhadap performa klasifikasi pada kondisi data latih terbatas (*small-data regime*):

| Kode | Kondisi Eksperimen | Arsitektur | Inisialisasi | Parameter Trainable | Target Analisis |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **B1** | From Scratch | ResNet-18 | Acak (*weights=None*) | 11.181.642 (100%) | Baseline tanpa prior visual |
| **B2** | Feature Extraction | ResNet-18 | Pretrained ImageNet | 5.130 (~0,05%) | Stabilitas classifier head linier |
| **B3** | Progressive Fine-Tuning | ResNet-18 | Pretrained ImageNet | 8.398.858 (~75%) | Unfreezing layer4 + differential LR |
| **B4a**| Tanpa Augmentasi | ResNet-18 | Pretrained ImageNet | 8.398.858 (~75%) | Kontrol dasar untuk augmentasi |
| **B4b**| Dengan Augmentasi | ResNet-18 | Pretrained ImageNet | 8.398.858 (~75%) | Regularisasi stokastik (Crop, Flip, Jitter) |
| **B5a**| EfficientNet-B0 | EfficientNet-B0 | Pretrained ImageNet | 424.970 (~8%) | Compound scaling & efisiensi parameter |
| **B5b**| MobileNetV3-Small | MobileNetV3-Small | Pretrained ImageNet | 601.098 (~24%) | Arsitektur lightweight untuk perangkat edge |

---

## 2. Struktur Repositori

```text
├── gslc.ipynb                  # Notebook utama yang memuat kode, visualisasi, dan teks analisis
├── build_notebook.py           # Skrip otomatisasi untuk merekonstruksi notebook gslc.ipynb
├── requirements.txt            # Daftar pustaka dependensi Python
├── README.md                   # Dokumentasi proyek
├── data/                       # Direktori dataset Oxford-IIIT Pet (diunduh otomatis oleh torchvision)
│   └── oxford-iiit-pet/
└── logs/                       # Hasil output empiris, metrik, dan visualisasi
    ├── C1_consolidated_results.csv
    ├── C1_comparison_curves.png
    ├── C1_bar_comparison.png
    ├── B4_augmentation_comparison.png
    ├── C3_failure_analysis.png
    ├── C5_gradcam_comparison.png
    ├── *_history.json
    ├── *_eval.csv
    ├── curve_*.png
    └── cm_*.png
```

---

## 3. Ringkasan Hasil Empiris (Test Set)

Hasil evaluasi pada test set independen (10 kelas, 300 citra uji) dirangkum dalam tabel berikut:

| Eksperimen | Arsitektur | Train Time (s) | Epochs Converge | Val Acc Terbaik | Test Accuracy | Macro F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **B1 - From Scratch** | ResNet-18 | 211.3 | 15 (Early Stop) | 0.3490 | 0.3467 | 0.3254 |
| **B2 - Feature Extraction** | ResNet-18 | 459.0 | 36 | 0.9329 | 0.9200 | 0.9214 |
| **B3 - Progressive Fine-Tuning** | ResNet-18 | 224.1 | 28 | 0.9530 | **0.9467** | **0.9471** |
| **B4a - Tanpa Augmentasi** | ResNet-18 | 224.1 | 28 | 0.9530 | **0.9467** | **0.9471** |
| **B4b - Dengan Augmentasi** | ResNet-18 | 289.8 | 15 (Early Stop) | 0.9597 | 0.9067 | 0.9063 |
| **B5a - EfficientNet-B0** | EfficientNet-B0 | 454.2 | 50 | **0.9732** | 0.9200 | 0.9211 |
| **B5b - MobileNetV3-Small** | MobileNetV3-Small | **52.2** | 7 (Early Stop) | 0.9060 | 0.8733 | 0.8718 |

### Temuan Kunci
1. **Disparitas Ekstrem (B1 vs B2/B3):** Pelatihan dari nol (*from scratch*) mengalami *severe overfitting* dengan Macro F1 hanya 0.3254 akibat rasio data terhadap parameter yang sangat kecil (0,00012 sampel/parameter). Transfer learning langsung mendongkrak performa melampaui 92%.
2. **Keunggulan Progressive Fine-Tuning (B3):** Membuka `layer4` dengan *differential learning rate* menghasilkan Test Macro F1 tertinggi (0.9471), mengungguli Feature Extraction murni (0.9214) tanpa memicu *catastrophic forgetting*.
3. **Over-regularization pada Augmentasi (B4b):** Augmentasi agresif (khususnya *ColorJitter*) meningkatkan *Validation Accuracy* (0.9597), namun menurunkan *Test Macro F1* (0.9063) karena mengaburkan corak dan pola warna bulu halus khas ras murni.
4. **Efisiensi Edge (B5b):** MobileNetV3-Small hanya memerlukan 52,2 detik pelatihan dan 601 ribu parameter untuk mencapai Macro F1 0.8718, menjadikannya pilihan optimal untuk inferensi berdaya rendah.

---

## 4. Instalasi dan Persyaratan Lingkungan

### Prasyarat Sistem
* Python 3.10 atau lebih baru (direkomendasikan menggunakan Conda environment).
* Perangkat GPU dengan akselerasi CUDA (diuji pada NVIDIA GeForce RTX 5050 Laptop GPU, CUDA 12.8 / PyTorch 2.11).

### Langkah Instalasi
1. Klon atau buka repositori ini di direktori kerja lokal.
2. Buat dan aktifkan virtual environment (opsional namun disarankan):
   ```bash
   conda create -n mopad_gpu python=3.11 -y
   conda activate mopad_gpu
   ```
3. Pasang seluruh dependensi yang dibutuhkan:
   ```bash
   pip install -r requirements.txt
   ```

---

## 5. Cara Menjalankan Eksperimen

1. Buka Jupyter Notebook:
   ```bash
   jupyter notebook gslc.ipynb
   ```
   atau buka file `gslc.ipynb` secara langsung di Visual Studio Code dengan kernel Python yang sesuai.
2. Jalankan sel notebook secara berurutan (*Run All Cells* atau eksekusi sel per bagian).
3. Dataset Oxford-IIIT Pet akan diunduh secara otomatis ke folder `./data/` pada saat sel Bagian A dieksekusi.
4. Seluruh grafik evaluasi, kurva pembelajaran, matriks konfusi, dan berkas log CSV/JSON akan secara otomatis tersimpan di dalam folder `./logs/`.

---

## 6. Referensi Literatur Utama

* **He, K., Zhang, X., Ren, S., & Sun, J. (2016).** *Deep residual learning for image recognition.* CVPR, 770–778.
* **Howard, J., & Ruder, S. (2018).** *Universal language model fine-tuning for text classification.* ACL, 328–339.
* **Kirkpatrick, J., et al. (2017).** *Overcoming catastrophic forgetting in neural networks.* PNAS, 114(13), 3521–3526.
* **Yosinski, J., Clune, J., Bengio, Y., & Lipson, H. (2014).** *How transferable are features in deep neural networks?* NeurIPS, 27, 3320–3328.
* **Shorten, C., & Khoshgoftaar, T. M. (2019).** *A survey on image data augmentation for deep learning.* Journal of Big Data, 6(1), 1–48.
* **Selvaraju, R. R., et al. (2017).** *Grad-CAM: Visual explanations from deep networks via gradient-based localization.* ICCV, 618–626.
