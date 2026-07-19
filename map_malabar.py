"""
Script untuk mapping Malabar_Dataset ke struktur segar/layu/busuk
"""
import os
import shutil
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MALABAR_DIR = os.path.join(BASE_DIR, 'malabar_dataset')
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')

# Mapping: folder sumber → target class
MAPPING = {
    'Healthy-Leaf(1399)': 'segar',
    'Pest-Damage(513)': 'layu',
    'Anthracnose(102)': 'busuk',
    'Bacterial-Spot(752)': 'busuk',
    'Downy-Mildew(240)': 'busuk',
}

def map_dataset():
    """Copy images dari malabar_dataset ke dataset/segar|layu|busuk"""
    
    # Buat folder target jika belum ada
    for kelas in ['segar', 'layu', 'busuk']:
        kelas_dir = os.path.join(DATASET_DIR, kelas)
        os.makedirs(kelas_dir, exist_ok=True)
    
    print("=" * 70)
    print("MAPPING MALABAR_DATASET KE DATASET STRUKTUR")
    print("=" * 70)
    
    total_copied = 0
    
    for src_folder, target_class in MAPPING.items():
        src_path = os.path.join(MALABAR_DIR, src_folder)
        target_dir = os.path.join(DATASET_DIR, target_class)
        
        if not os.path.exists(src_path):
            print(f"\n⚠️  {src_folder} tidak ditemukan!")
            continue
        
        # Hitung file
        files = [f for f in os.listdir(src_path) if os.path.isfile(os.path.join(src_path, f))]
        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        
        if not image_files:
            print(f"\n⚠️  {src_folder} tidak ada image!")
            continue
        
        print(f"\n📁 {src_folder} ({len(image_files)} images)")
        print(f"   → dataset/{target_class}/")
        
        # Copy semua gambar
        copied = 0
        for img_file in image_files:
            src_file = os.path.join(src_path, img_file)
            dst_file = os.path.join(target_dir, img_file)
            
            # Avoid overwrite - add prefix jika sudah ada
            if os.path.exists(dst_file):
                name, ext = os.path.splitext(img_file)
                counter = 1
                while os.path.exists(os.path.join(target_dir, f"{name}_{counter}{ext}")):
                    counter += 1
                dst_file = os.path.join(target_dir, f"{name}_{counter}{ext}")
            
            try:
                shutil.copy2(src_file, dst_file)
                copied += 1
            except Exception as e:
                print(f"   ❌ Error copy {img_file}: {e}")
        
        print(f"   ✅ Copied: {copied} images")
        total_copied += copied
    
    print("\n" + "=" * 70)
    print(f"✅ SELESAI! Total {total_copied} images dicopy ke dataset/")
    print("=" * 70)
    
    # Summary
    print("\n[Dataset Summary]")
    for kelas in ['segar', 'layu', 'busuk']:
        kelas_dir = os.path.join(DATASET_DIR, kelas)
        count = len([f for f in os.listdir(kelas_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))])
        print(f"  dataset/{kelas}/: {count} images")

if __name__ == '__main__':
    map_dataset()
