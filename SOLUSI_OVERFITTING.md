# 🎯 PANDUAN MENGATASI OVERFITTING & BIAS KE BUSUK

## 📊 Status Dataset Anda
✅ **Dataset: BALANCED** (300:300:300)  
✅ **Ukuran: ADEQUATE** (900 total images)  
✅ **Fitur Ekstraksi: COMPREHENSIVE** (16 features)

---

## 🔴 MASALAH POTENSIAL

### 1. **Fitur Brown/Coklat Bias ke Busuk**
```python
# Dalam ekstraksi_fitur_glcm(), fitur ini:
lower_brown = np.array([10, 40, 40])      # HSV range coklat
upper_brown = np.array([30, 255, 200])
brown_ratio = cv2.inRange(hsv, lower_brown, upper_brown)
```
**Masalah**: Fitur ini SANGAT mengidentifikasi busuk (coklat = busuk).  
**Solusi**: 
- Pertahankan, tapi normalize lebih baik
- Tambah fitur pembeda untuk layu vs busuk

### 2. **Feature Normalization**
```python
# Saat ini: StandardScaler (baik)
# Tapi: Fitur edge/brown punya range berbeda
```
**Masalah**: `brown_ratio` (0-1) vs `edge_density` (0-1) vs HSV mean (0-255)  
**Status**: ✅ StandardScaler sudah menangani ini

### 3. **Model Complexity vs Data Size**
- SVM RBF dengan 16 features + 900 samples = **OK, tidak overfit**
- Tapi bisa overfitting jika `C` terlalu besar

---

## ✅ SOLUSI YANG SUDAH DITERAPKAN

### 1. ✔️ Class Weight Balanced
```python
SVC(kernel='rbf', class_weight='balanced', ...)
```
**Efek**: Memberikan penalty lebih tinggi untuk salah klasifikasi class minoritas

### 2. ✔️ F1-Weighted Scoring
```python
scoring='f1_weighted'  # Bukan 'accuracy'
```
**Efek**: Mengutamakan recall & precision per-class, bukan just accuracy

### 3. ✔️ Stratified K-Fold
```python
train_test_split(..., stratify=y)
GridSearchCV(..., cv=5)  # Stratified K-Fold
```
**Efek**: Menjaga proporsi kelas di train/test

---

## 🚀 IMPROVEMENT RECOMMENDATIONS

### A. FITUR ENGINEERING (Tambahan)
**Tambah fitur spesifik untuk membedakan layu vs busuk:**

```python
# Tambah di ekstraksi_fitur_glcm()

# 1. Green channel intensity (segar = hijau tinggi)
green = img_resized[:,:,1]  # BGR channel, green adalah index 1
green_intensity = np.mean(green)

# 2. Color saturation variance (busuk = saturation tinggi & tidak uniform)
saturation = hsv[:,:,1]
sat_variance = np.std(saturation)

# 3. Texture uniformity (segar = uniform, busuk = berantakan)
texture_uniformity = energy  # dari GLCM

# 4. Yellow hue presence (layu = lebih kuning)
lower_yellow = np.array([15, 50, 50])
upper_yellow = np.array([35, 255, 255])
yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
yellow_ratio = np.sum(yellow_mask > 0) / (256 * 256)

fitur_final = fitur_warna + fitur_glcm + fitur_edge + \
              [green_intensity, sat_variance, texture_uniformity, yellow_ratio]
```

### B. MONITORING OVERFITTING
**Ukuran yang harus dimonitor:**

```
TRAIN SET           TEST SET              DIAGNOSIS
ACC: 95%   F1: 0.94 | ACC: 93%   F1: 0.92 ✅ Normal
ACC: 98%   F1: 0.97 | ACC: 85%   F1: 0.83 ❌ OVERFITTING!

BUSUK Recall: 0.95  | LAYU Recall: 0.75   ❌ BIAS (busuk bias)
BUSUK Recall: 0.88  | LAYU Recall: 0.87   ✅ BALANCED
```

### C. HYPERPARAMETER TUNING
**Saat ini GridSearch mencari:**
```python
param_grid = {
    'C'    : [0.1, 1, 10, 15, 100],      # Regularization strength
    'gamma': ['scale', 0.01, 0.1, 1]    # Kernel coefficient
}
```

**Rekomendasi untuk cegah overfitting:**
```python
# Kurangi C (lebih strong regularization)
'C': [0.01, 0.1, 1, 10],  # Lebih konservatif

# Lebih fine-grained gamma
'gamma': ['scale', 0.001, 0.01, 0.1, 1]
```

### D. STRATEGI TRAINING LANJUTAN

```python
# 1. Cross-Validation Report per-fold
# 2. ROC-AUC per class (bukan just accuracy)
# 3. Confusion Matrix analysis
# 4. Feature importance analysis (SVM koef)
```

---

## 📋 CHECKLIST IMPLEMENTASI

- [ ] **Jalankan script training**
  ```bash
  # Buka Flask app atau run training_model()
  ```

- [ ] **Monitor Output**
  ```
  [Dataset Info] - Lihat distribusi kelas
  [Grid Search Results] - Lihat parameter terbaik
  [Per-Class Performance] - PENTING! Cek recall/precision
  ```

- [ ] **Analisa Per-Class Metrics**
  ```
  BUSUK Precision/Recall tinggi → Bias ke busuk
  LAYU Precision/Recall rendah  → Kurang terekstraksi
  SEGAR Precision/Recall stabil → Baseline good
  ```

- [ ] **Jika ada bias busuk:**
  1. Tambah fitur yellow_ratio untuk layu
  2. Turunkan C di SVM (lebih regularization)
  3. Lihat fitur mana paling penting untuk busuk

---

## 🔍 CARA CEK HASIL

**File yang di-generate saat training:**
```
model_svm.pkl      - Model terlatih
scaler.pkl         - Feature scaler
```

**Test dengan gambar real:**
```bash
# Upload gambar ke /predict
# Lihat detail_proba di result.html
```

**Metrics yang harus diperhatikan:**
```
❌ Jangan: "Overall Accuracy: 90%"
✅ Ya: "BUSUK F1: 0.92, LAYU F1: 0.89, SEGAR F1: 0.93"
```

---

## 🛠️ TOOLS UNTUK ANALISIS LANJUTAN

```bash
# Jalankan script analisis dataset
python analyze_dataset.py

# Untuk training verbose
# Output akan menunjukkan:
# - Class distribution
# - CV scores per fold
# - Per-class metrics
```

---

## 📚 SUMMARY SOLUSI

| Problem | Root Cause | Solution | Status |
|---------|-----------|----------|--------|
| Bias Busuk | Feature coklat terlalu diskriminatif | Tambah yellow_ratio | ⏳ TODO |
| Overfitting | C terlalu besar | Turunkan C range | ⏳ TODO |
| Imbalance | Dataset imbalanced | class_weight='balanced' | ✅ DONE |
| Metric | Hanya lihat accuracy | F1-weighted scoring | ✅ DONE |
| Cross-Val | Random split | Stratified K-Fold | ✅ DONE |

---

## 🎓 NEXT STEPS

1. **Jalankan training** dengan perubahan baru (class_weight + F1-scoring)
2. **Catat hasil** per-class metrics untuk tiap kelas
3. **Jika BUSUK F1 >> others**, berarti ada bias → tambah fitur
4. **Jika LAYU F1 << others**, berarti layu kurang terekstraksi → improve features
5. **Iterate** sampai semua F1-score seimbang (±0.02)

Semoga membantu! 🚀
