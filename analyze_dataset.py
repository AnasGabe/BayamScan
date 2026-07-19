"""
Script untuk analisis dataset dan deteksi overfitting/bias
Gunakan: python analyze_dataset.py
"""

import os
import cv2
import numpy as np
from collections import Counter
import json

DATASET_PATH = 'dataset'
KELAS = ['segar', 'layu', 'busuk']


def count_dataset():
    """Hitung jumlah gambar per kelas"""
    print("=" * 60)
    print("ANALISIS DISTRIBUSI DATASET")
    print("=" * 60)
    
    class_counts = {}
    total_images = 0
    
    for kelas in KELAS:
        folder = os.path.join(DATASET_PATH, kelas)
        if os.path.exists(folder):
            files = [f for f in os.listdir(folder) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            count = len(files)
            class_counts[kelas] = count
            total_images += count
            percentage = (count / total_images * 100) if total_images > 0 else 0
            print(f"\n📊 {kelas.upper()}: {count} images ({percentage:.1f}%)")
        else:
            class_counts[kelas] = 0
            print(f"\n❌ {kelas.upper()}: Folder tidak ditemukan!")
    
    print("\n" + "=" * 60)
    print("ANALISIS KESEIMBANGAN (CLASS IMBALANCE)")
    print("=" * 60)
    
    if total_images == 0:
        print("❌ Tidak ada gambar dalam dataset!")
        return
    
    counts = list(class_counts.values())
    max_count = max(counts)
    min_count = min(counts)
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    
    print(f"\n📈 Total images: {total_images}")
    print(f"📈 Max class size: {max_count} ({class_counts[KELAS[counts.index(max_count)]]}) images")
    print(f"📉 Min class size: {min_count} ({class_counts[KELAS[counts.index(min_count)]]}) images")
    print(f"⚖️  Imbalance Ratio: {imbalance_ratio:.2f}x")
    
    if imbalance_ratio < 1.2:
        print("   ✅ BAIK: Dataset cukup seimbang")
    elif imbalance_ratio < 2.0:
        print("   ⚠️  MODERATE: Ada sedikit imbalance, pantau performa per-class")
    else:
        print("   ❌ SEVERE: Imbalance signifikan, gunakan class_weight!")
    
    print("\n" + "=" * 60)
    print("REKOMENDASI MITIGASI")
    print("=" * 60)
    
    if imbalance_ratio > 1.5:
        print("\n1️⃣  Gunakan class_weight='balanced' di SVM ✅ (sudah diterapkan)")
        print("   - Memberikan bobot lebih tinggi ke class yang lebih kecil")
        
    print("\n2️⃣  Cross-Validation Strategy:")
    print("   - Gunakan Stratified K-Fold (✅ sudah diterapkan)")
    print("   - Ukur dengan F1-Score, bukan Accuracy")
    
    print("\n3️⃣  Data Augmentation (jika dataset kecil):")
    if total_images < 500:
        print("   ❌ DIPERLUKAN! Dataset < 500 images")
        print("   - Rotation, flip, brightness adjustment")
        print("   - Geometric transformations")
    else:
        print("   ✅ Dataset sudah cukup besar")
    
    # Cek fitur ekstraksi
    print("\n4️⃣  Feature Engineering:")
    print("   - HSV features: Baik untuk warna busuk/segar/layu ✅")
    print("   - GLCM features: Baik untuk tekstur permukaan ✅")
    print("   - Edge features: Review fitur brown_ratio (mungkin bias)")
    
    print("\n5️⃣  Model Tuning:")
    print("   - GridSearchCV dengan F1-weighted (✅ sudah diterapkan)")
    print("   - Monitor: Precision & Recall per-class")
    print("   - Hindari: Hanya perhatikan Overall Accuracy")


def check_image_quality():
    """Cek kualitas dan ukuran gambar"""
    print("\n" + "=" * 60)
    print("ANALISIS KUALITAS GAMBAR")
    print("=" * 60)
    
    for kelas in KELAS:
        folder = os.path.join(DATASET_PATH, kelas)
        if not os.path.exists(folder):
            continue
            
        files = [f for f in os.listdir(folder) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not files:
            print(f"\n⚠️  {kelas.upper()}: No images found")
            continue
        
        sizes = []
        dimensions = []
        
        for fname in files[:50]:  # Sample first 50
            try:
                path = os.path.join(folder, fname)
                img = cv2.imread(path)
                if img is not None:
                    h, w = img.shape[:2]
                    dimensions.append((w, h))
            except:
                pass
        
        if dimensions:
            dims = np.array(dimensions)
            print(f"\n📸 {kelas.upper()}:")
            print(f"   - Avg size: {dims[:, 0].mean():.0f}x{dims[:, 1].mean():.0f} px")
            print(f"   - Size range: {dims[:, 0].min()}-{dims[:, 0].max()}x{dims[:, 1].min()}-{dims[:, 1].max()}")


if __name__ == '__main__':
    count_dataset()
    check_image_quality()
    
    print("\n" + "=" * 60)
    print("✅ Analisis selesai!")
    print("=" * 60)
