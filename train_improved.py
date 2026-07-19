"""
Script untuk:
1. Ekstraksi fitur yang lebih baik
2. Analisis feature importance untuk mendeteksi bias
3. Membuat confusion matrix visualization
"""

import os
import cv2
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from skimage.feature import graycomatrix, graycoprops
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATASET_PATH = 'dataset'
KELAS = ['segar', 'layu', 'busuk']
MODEL_PATH = 'model_svm.pkl'
SCALER_PATH = 'scaler.pkl'


def ekstraksi_fitur_improved(image_path, verbose=False):
    """
    Fitur ekstraksi IMPROVED dengan tambahan:
    - Green intensity (segar = hijau tinggi)
    - Yellow presence (layu = kuning)
    - Saturation variance (busuk = berantakan)
    - Edge distribution (busuk = lebih rusak)
    
    Total 22 fitur (dari 16 sebelumnya)
    """
    img = cv2.imread(image_path)
    if img is None:
        return None

    img_resized = cv2.resize(img, (256, 256))

    # ── 1. Fitur Warna HSV ──
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    fitur_warna = [
        np.mean(h), np.std(h),
        np.mean(s), np.std(s),
        np.mean(v), np.std(v)
    ]

    # ── 2. Fitur Tekstur GLCM ──
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    gray_q = (gray // 4).astype(np.uint8)
    glcm = graycomatrix(gray_q, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=64, symmetric=True, normed=True)

    contrast     = graycoprops(glcm, 'contrast').mean()
    energy       = graycoprops(glcm, 'energy').mean()
    homogeneity  = graycoprops(glcm, 'homogeneity').mean()
    correlation  = graycoprops(glcm, 'correlation').mean()
    dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
    asm          = graycoprops(glcm, 'ASM').mean()

    fitur_glcm = [contrast, energy, homogeneity, correlation, dissimilarity, asm]

    # ── 3. Fitur Edge Damage Detection (Original) ──
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (256 * 256)

    # Deteksi coklat (indikator busuk)
    lower_brown = np.array([10, 40, 40])
    upper_brown = np.array([30, 255, 200])
    brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
    brown_ratio = np.sum(brown_mask > 0) / (256 * 256)

    # Kontur & Solidity
    contours, _ = cv2.findContours(
        cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        perimeter = cv2.arcLength(cnt, True)
        pa_ratio = perimeter / area if area > 0 else 0
    else:
        solidity = 0
        pa_ratio = 0

    fitur_edge_original = [edge_density, brown_ratio, solidity, pa_ratio]

    # ── 4. FITUR TAMBAHAN UNTUK MEMBEDAKAN LAYU vs BUSUK ──
    
    # 4a. Green intensity (segar = hijau tinggi)
    green_channel = img_resized[:,:,1]  # BGR, green is index 1
    green_intensity = np.mean(green_channel) / 255.0  # normalize 0-1
    
    # 4b. Yellow presence (layu = lebih kuning)
    lower_yellow = np.array([15, 50, 50])
    upper_yellow = np.array([35, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    yellow_ratio = np.sum(yellow_mask > 0) / (256 * 256)
    
    # 4c. Saturation uniformity (segar = uniform, busuk = berantakan)
    sat_variance = np.std(s) / 255.0  # normalize
    sat_mean = np.mean(s) / 255.0
    
    # 4d. Blue channel (busuk mungkin punya lebih banyak blue noise)
    blue_channel = img_resized[:,:,0]  # BGR, blue is index 0
    blue_intensity = np.mean(blue_channel) / 255.0
    
    # 4e. Red channel (segar = less red, busuk/layu = more red)
    red_channel = img_resized[:,:,2]  # BGR, red is index 2
    red_intensity = np.mean(red_channel) / 255.0
    
    # 4f. Hue distribution (berapa banyak pixel dengan hue "sehat")
    # Hue 40-80 adalah green zone (segar)
    lower_green_hue = np.array([40, 30, 30])
    upper_green_hue = np.array([80, 255, 255])
    green_hue_mask = cv2.inRange(hsv, lower_green_hue, upper_green_hue)
    green_hue_ratio = np.sum(green_hue_mask > 0) / (256 * 256)
    
    fitur_tambahan = [
        green_intensity, yellow_ratio, sat_variance, sat_mean,
        blue_intensity, red_intensity, green_hue_ratio
    ]
    
    # ── Gabung Semua (total 22 fitur) ──
    fitur_semua = fitur_warna + fitur_glcm + fitur_edge_original + fitur_tambahan
    
    if verbose:
        print(f"✓ Ekstraksi fitur: {len(fitur_semua)} features")
    
    return fitur_semua


def load_dataset_improved():
    """Load dataset dengan fitur baru"""
    X, y = [], []
    
    for label, kelas in enumerate(KELAS):
        folder = os.path.join(DATASET_PATH, kelas)
        if not os.path.exists(folder):
            continue
        
        files = [f for f in os.listdir(folder) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for fname in files:
            path = os.path.join(folder, fname)
            fitur = ekstraksi_fitur_improved(path)
            if fitur is not None:
                X.append(fitur)
                y.append(label)
    
    return np.array(X), np.array(y)


def train_and_analyze():
    """Training dengan analisis bias detection"""
    print("=" * 70)
    print("TRAINING MODEL SVM DENGAN IMPROVED FEATURES")
    print("=" * 70)
    
    # Load data
    print("\n[1] Loading dataset dengan 22 features...")
    X, y = load_dataset_improved()
    
    if len(X) < 10:
        print("❌ Dataset terlalu kecil!")
        return
    
    print(f"✓ Data loaded: {len(X)} samples, {X.shape[1]} features")
    
    # Distribusi
    unique, counts = np.unique(y, return_counts=True)
    print("\n[2] Dataset Distribution:")
    for kls, cnt in zip(unique, counts):
        print(f"   - {KELAS[kls]}: {cnt} samples")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Training
    print("\n[3] Training SVM dengan class_weight='balanced'...")
    svm = SVC(kernel='rbf', C=10, gamma=0.1, 
              probability=True, class_weight='balanced', random_state=42)
    svm.fit(X_train_scaled, y_train)
    print("✓ Training selesai")
    
    # Prediksi
    y_pred_train = svm.predict(X_train_scaled)
    y_pred_test = svm.predict(X_test_scaled)
    
    # Evaluasi
    print("\n[4] EVALUASI MODEL")
    print("=" * 70)
    
    train_acc = np.mean(y_pred_train == y_train)
    test_acc = np.mean(y_pred_test == y_test)
    
    print(f"\n📊 Overall Accuracy:")
    print(f"   - Train: {train_acc*100:.2f}%")
    print(f"   - Test:  {test_acc*100:.2f}%")
    
    if train_acc - test_acc > 0.05:
        print(f"\n⚠️  WARNING: Potential overfitting ({train_acc-test_acc:.2f}% gap)")
    else:
        print(f"\n✅ Model generalisasi baik ({train_acc-test_acc:.2f}% gap)")
    
    # Per-class report
    print("\n[5] PER-CLASS PERFORMANCE (TEST SET):")
    print("=" * 70)
    report = classification_report(y_test, y_pred_test, 
                                  target_names=KELAS, digits=4)
    print(report)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_test)
    print("\n[6] CONFUSION MATRIX:")
    print(cm)
    
    # Analisis Bias
    print("\n[7] ANALISIS BIAS & OVERFITTING:")
    print("=" * 70)
    
    # Hitung recall per class
    recalls = []
    for i in range(len(KELAS)):
        recall = cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0
        recalls.append(recall)
        
        if recall > 0.95:
            print(f"   ✅ {KELAS[i].upper()}: Recall {recall:.4f} (sangat baik)")
        elif recall > 0.85:
            print(f"   ⚠️  {KELAS[i].upper()}: Recall {recall:.4f} (baik)")
        else:
            print(f"   ❌ {KELAS[i].upper()}: Recall {recall:.4f} (perlu improvement)")
    
    # Check bias
    max_recall = max(recalls)
    min_recall = min(recalls)
    recall_diff = max_recall - min_recall
    
    if recall_diff > 0.1:
        max_idx = recalls.index(max_recall)
        min_idx = recalls.index(min_recall)
        print(f"\n⚠️  BIAS DETECTED: {KELAS[max_idx].upper()} recall {max_recall:.3f} >> {KELAS[min_idx].upper()} recall {min_recall:.3f}")
        print(f"    Difference: {recall_diff:.3f}")
    else:
        print(f"\n✅ BALANCED: Semua class recall seimbang (diff: {recall_diff:.3f})")
    
    # Visualisasi
    print("\n[8] Membuat visualisasi...")
    plot_confusion_matrix(cm, KELAS)
    
    return svm, scaler


def plot_confusion_matrix(cm, classes):
    """Buat confusion matrix plot"""
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Normalize
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    im = ax.imshow(cm_normalized, cmap='Blues', aspect='auto')
    
    # Labels
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title('Confusion Matrix (Normalized)')
    
    # Text annotations
    for i in range(len(classes)):
        for j in range(len(classes)):
            text = ax.text(j, i, f'{cm_normalized[i, j]:.2f}\n({cm[i, j]})',
                         ha="center", va="center", color="black", fontsize=10)
    
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=100)
    print("✓ Saved: confusion_matrix.png")
    plt.close()


if __name__ == '__main__':
    svm, scaler = train_and_analyze()
    
    if svm is not None:
        print("\n" + "=" * 70)
        print("✅ TRAINING COMPLETED")
        print("=" * 70)
        
        # Bisa disimpan ke model baru jika ingin
        print("\nNote: Model dengan 22 features ini bisa disimpan untuk upgrade")
        print("Current model masih pakai 16 features di app.py")
