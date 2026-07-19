# Laporan Proyek BayamScan

## 1. Judul
Klasifikasi Kesegaran Daun Bayam Berbasis Citra Digital Menggunakan GLCM dan SVM

## 2. Latar Belakang
Daun bayam memiliki ciri visual yang berbeda antara kondisi segar, layu, dan busuk. Dengan memanfaatkan pengolahan citra digital, sistem dapat mengenali kondisi daun berdasarkan warna, tekstur, dan bentuk permukaan secara otomatis. Metode yang digunakan dalam proyek ini adalah:

- Ekstraksi fitur GLCM untuk mencirikan tekstur daun
- Ekstraksi fitur warna HSV/RGB untuk mencirikan warna daun
- Klasifikasi menggunakan Support Vector Machine (SVM)

## 3. Tujuan
Tujuan dari proyek ini adalah:

1. Membangun aplikasi web untuk mendeteksi kesegaran daun bayam berdasarkan citra.
2. Menggunakan metode GLCM dan SVM sebagai inti klasifikasi.
3. Menyediakan fitur pelatihan model dari dataset yang dapat dipilih pengguna.
4. Menampilkan evaluasi model seperti akurasi, precision, recall, dan F1-score.

## 4. Metode yang Digunakan
### 4.1 Ekstraksi Fitur
Fitur diekstraksi dari citra daun bayam melalui beberapa tahapan:

- Resize citra menjadi ukuran 256x256
- Ekstraksi fitur warna HSV dan/atau RGB
- Ekstraksi fitur tekstur GLCM
- Ekstraksi fitur tepi dan bentuk untuk mendukung klasifikasi

### 4.2 Klasifikasi
Model yang digunakan adalah SVM dengan beberapa pilihan kernel:

- RBF
- Linear
- Polynomial
- Sigmoid

Proses training dilakukan menggunakan GridSearchCV untuk mencari parameter terbaik.

## 5. Struktur Aplikasi
Proyek ini terdiri dari beberapa bagian utama:

- `app.py` : file utama aplikasi Flask
- `templates/` : halaman web untuk upload, training, hasil prediksi, dan informasi
- `static/upload/` : folder penyimpanan gambar upload pengguna
- `dataset/` : folder dataset untuk kelas segar, layu, dan busuk
- `map_malabar.py` : script untuk memetakan dataset Malabar ke struktur dataset yang dipakai aplikasi

## 6. Alur Kerja Aplikasi
### 6.1 Training Model
1. User memilih path dataset.
2. User memilih kernel SVM dan mode warna.
3. Sistem membaca gambar dari folder dataset.
4. Sistem mengekstraksi fitur dari setiap gambar.
5. Sistem melakukan training menggunakan SVM.
6. Sistem menghitung evaluasi model dan menampilkan hasilnya.

### 6.2 Prediksi Gambar
1. User mengupload gambar daun bayam.
2. Sistem mengekstraksi fitur dari gambar input.
3. Sistem menerapkan model SVM yang sudah dilatih.
4. Sistem mengeluarkan label kelas dan persentase kesegaran.
5. Hasil ditampilkan pada halaman hasil prediksi.

## 7. Fitur yang Sudah Ditambahkan
Beberapa fitur yang telah diimplementasikan antara lain:

- Pilihan path dataset custom
- Pilihan kernel SVM (RBF, Linear, Polynomial, Sigmoid)
- Pilihan mode ekstraksi warna (HSV, RGB, RGB+HSV)
- Evaluasi model dengan akurasi, precision, recall, dan F1-score
- Penyimpanan metadata model untuk prediksi konsisten
- Dukungan mapping dataset dari folder Malabar ke struktur dataset aplikasi

## 8. Hasil yang Diharapkan
Aplikasi ini diharapkan mampu:

- Mengklasifikasikan daun bayam ke dalam kelas segar, layu, atau busuk.
- Memberikan hasil prediksi secara cepat melalui antarmuka web.
- Menyediakan informasi evaluasi model bagi pengguna yang ingin menganalisis performa sistem.

## 9. Cara Menjalankan
### Instalasi dependensi
```bash
pip install -r requirements.txt
```

### Jalankan aplikasi
```bash
python app.py
```

### Opsional: mapping dataset Malabar
```bash
python map_malabar.py
```

## 10. Kesimpulan
Proyek BayamScan berhasil dikembangkan sebagai aplikasi web berbasis Flask yang mampu melakukan klasifikasi kesegaran daun bayam menggunakan metode GLCM dan SVM. Aplikasi ini memiliki fitur pelatihan model yang fleksibel, evaluasi model yang lengkap, serta kemampuan prediksi dari gambar upload pengguna.

## 11. Saran Pengembangan
Untuk pengembangan lebih lanjut, sistem dapat ditingkatkan dengan:

- Menambahkan data augmentation untuk memperbaiki generalisasi model
- Menggunakan model yang lebih modern seperti CNN
- Menambah fitur visualisasi confusion matrix
- Mengoptimalkan fitur ekstraksi agar lebih robust terhadap variasi latar belakang dan pencahayaan
