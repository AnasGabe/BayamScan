# 🌿 BayamScan — Klasifikasi Daun Bayam (GLCM + SVM)

Aplikasi web Flask untuk mengklasifikasikan kesegaran daun bayam menggunakan metode ekstraksi fitur **GLCM (Gray Level Co-occurrence Matrix)** dan klasifikasi **SVM (Support Vector Machine)**.

---

## 📂 Struktur Proyek

```
bayam_app/
│
├── dataset/
│   ├── segar/        ← Masukkan gambar daun segar di sini
│   ├── layu/         ← Masukkan gambar daun layu di sini
│   └── busuk/        ← Masukkan gambar daun busuk di sini
│
├── static/
│   └── upload/       ← Gambar yang diupload user (otomatis)
│
├── templates/
│   ├── index.html    ← Halaman utama (upload + training)
│   ├── result.html   ← Halaman hasil prediksi
│   └── info.html     ← Halaman penjelasan GLCM & SVM
│
├── app.py            ← File utama Flask
├── map_malabar.py    ← Script mapping Malabar_Dataset ke segar/layu/busuk
├── requirements.txt  ← Daftar library
└── README.md
```

---

## ⚙️ Cara Instalasi & Menjalankan

### 1. Install Library
```bash
pip install -r requirements.txt
```

### 2. Siapkan Dataset
Masukkan gambar daun bayam ke folder yang sesuai:
- `dataset/segar/` → gambar daun segar (min. 50 gambar)
- `dataset/layu/`  → gambar daun layu  (min. 50 gambar)
- `dataset/busuk/` → gambar daun busuk (min. 50 gambar)

Atau gunakan folder dataset lain dengan menuliskannya pada field "Path Dataset" saat menekan tombol training.

Contoh: `dataset` atau `malabar_dataset` (jika sudah dimapping ke struktur yang benar).

Format gambar: PNG, JPG, atau JPEG

### 3. Jalankan Aplikasi
```bash
python app.py
```

### 4. Buka Browser
Akses di: **http://127.0.0.1:5000**

### 5. Mapping Malabar Dataset (opsional)
Jika Anda memiliki `malabar_dataset/` dan ingin memetakan ke kelas segar/layu/busuk, jalankan:
```bash
python map_malabar.py
```
Script ini akan meng-copy file ke folder `dataset/segar`, `dataset/layu`, dan `dataset/busuk`.

---

## 🚀 Cara Penggunaan

1. **Isi Path Dataset** (opsional) → misalnya `dataset` atau `malabar_dataset`
2. **Pilih Kernel SVM** → `rbf`, `linear`, `poly`, `sigmoid`
3. **Pilih Color Mode** → `HSV`, `RGB`, atau `RGB+HSV`
4. **Latih Model** → Klik tombol "⚙️ Mulai Training"
5. **Upload Gambar** → Pilih atau drag-drop foto daun bayam
6. **Lihat Hasil** → Sistem akan menampilkan kelas dan persentase kesegaran

---

## 🔬 Metode

### Ekstraksi Fitur
Aplikasi mengekstraksi fitur gabungan warna + tekstur + bentuk:

- Warna HSV: mean & std untuk H, S, V
- Warna RGB: mean & std untuk R, G, B
- Tekstur GLCM: Contrast, Energy, Homogeneity, Correlation, Dissimilarity, ASM
- Edge Damage: Edge Density, Brown Ratio, Solidity, PA Ratio

Secara default aplikasi menggunakan HSV, tetapi Anda dapat memilih:
- `HSV`
- `RGB`
- `RGB+HSV`

### GLCM
- Jarak: 1 piksel
- Sudut: 0°, 45°, 90°, 135°
- Level yang digunakan: 64 (gambar di-quantize dari 256 ke 64 level)

### SVM
- Kernel: pilihan `rbf`, `linear`, `poly`, `sigmoid`
- Parameter dicari dengan `GridSearchCV`
- Preprocessing: `StandardScaler`

---

## 📊 Output

- **Label Kelas**: Segar / Layu / Busuk
- **Persentase Kesegaran**: 0–100% (Segar=100%, Layu=50%, Busuk=10%)
- **Distribusi Probabilitas**: Probabilitas tiap kelas
- **Evaluasi Training**: akurasi, precision, recall, f1-score, CV F1-score
- **Model Metadata**: kernel dan color mode yang dipakai
- **Rekomendasi**: Saran penanganan daun
