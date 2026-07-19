import json
import os
import cv2
import numpy as np
import pickle
from flask import Flask, render_template, request, redirect, url_for
from skimage.feature import graycomatrix, graycoprops
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/upload'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DEFAULT = os.path.join(BASE_DIR, 'dataset')
KELAS = ['segar', 'layu', 'busuk']
MODEL_PATH = os.path.join(BASE_DIR, 'model_svm.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'scaler.pkl')
MODEL_META_PATH = os.path.join(BASE_DIR, 'model_meta.json')
COLOR_MODES = ['hsv', 'rgb', 'rgb+hsv']
SVM_KERNELS = ['rbf', 'linear', 'poly', 'sigmoid']

# ─────────────────────────────────────────────
# FUNGSI EKSTRAKSI FITUR GLCM
# ─────────────────────────────────────────────

def resolve_dataset_path(dataset_path):
    if not dataset_path:
        return DATASET_DEFAULT
    if os.path.isabs(dataset_path):
        return dataset_path
    return os.path.join(BASE_DIR, dataset_path)


def extract_color_features(img, color_mode='hsv'):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    features = []

    if color_mode in ('hsv', 'rgb+hsv'):
        features += [
            np.mean(h), np.std(h),
            np.mean(s), np.std(s),
            np.mean(v), np.std(v)
        ]

    if color_mode in ('rgb', 'rgb+hsv'):
        b, g, r = cv2.split(img)
        features += [
            np.mean(r), np.std(r),
            np.mean(g), np.std(g),
            np.mean(b), np.std(b)
        ]

    return features


def ekstraksi_fitur_glcm(image_path, color_mode='hsv'):
    """
    Ekstraksi fitur dari gambar daun bayam:
    - 6 fitur warna HSV dan/atau 6 fitur warna RGB
    - 6 fitur tekstur GLCM (Contrast, Energy, Homogeneity, Correlation, Dissimilarity, ASM)
    - 4 fitur deteksi bentuk/kerusakan tepi
    """
    if color_mode not in COLOR_MODES:
        color_mode = 'hsv'

    img = cv2.imread(image_path)
    if img is None:
        return None

    img_resized = cv2.resize(img, (256, 256))
    fitur_warna = extract_color_features(img_resized, color_mode)

    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    # Quantize piksel 0-255 -> 0-63 agar sesuai levels=64
    gray_q = (gray // 4).astype(np.uint8)
    glcm = graycomatrix(gray_q, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=64, symmetric=True, normed=True)

    contrast      = graycoprops(glcm, 'contrast').mean()
    energy        = graycoprops(glcm, 'energy').mean()
    homogeneity   = graycoprops(glcm, 'homogeneity').mean()
    correlation   = graycoprops(glcm, 'correlation').mean()
    dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
    asm           = graycoprops(glcm, 'ASM').mean()

    fitur_glcm = [contrast, energy, homogeneity, correlation, dissimilarity, asm]

    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (256 * 256)

    # Deteksi area coklat/kuning (indikator busuk/layu)
    lower_brown = np.array([10, 40, 40])
    upper_brown = np.array([30, 255, 200])
    brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
    brown_ratio = np.sum(brown_mask > 0) / (256 * 256)

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

    fitur_edge = [edge_density, brown_ratio, solidity, pa_ratio]
    return fitur_warna + fitur_glcm + fitur_edge


# ─────────────────────────────────────────────
# TRAINING MODEL SVM
# ─────────────────────────────────────────────

def training_model(dataset_path=None, kernel='rbf', color_mode='hsv'):
    dataset_path = resolve_dataset_path(dataset_path)
    if not os.path.isdir(dataset_path):
        print(f"[ERROR] Dataset path tidak ditemukan: {dataset_path}")
        return None, None, 0, 0

    X, y = [], []
    for label, kelas in enumerate(KELAS):
        folder = os.path.join(dataset_path, kelas)
        if not os.path.exists(folder):
            continue
        for fname in os.listdir(folder):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                path = os.path.join(folder, fname)
                fitur = ekstraksi_fitur_glcm(path, color_mode=color_mode)
                if fitur is not None:
                    X.append(fitur)
                    y.append(label)

    if len(X) < 10:
        return None, None, 0, 0

    X = np.array(X)
    y = np.array(y)

    # ── Analisis distribusi class ──
    unique, counts = np.unique(y, return_counts=True)
    print("\n[Dataset Info]")
    for kls, cnt in zip(unique, counts):
        print(f"  {KELAS[kls]}: {cnt} samples ({cnt/len(y)*100:.1f}%)")
    
    # Cek imbalance ratio
    max_count = np.max(counts)
    min_count = np.min(counts)
    imbalance_ratio = max_count / min_count
    print(f"  Imbalance Ratio: {imbalance_ratio:.2f}x")
    if imbalance_ratio > 2:
        print("  ⚠️ WARNING: Significant class imbalance detected!")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    param_grid = {'C': [0.1, 1, 10, 15, 100]}
    if kernel in ('rbf', 'poly', 'sigmoid'):
        param_grid['gamma'] = ['scale', 0.01, 0.1, 1]
    if kernel == 'poly':
        param_grid['degree'] = [2, 3]

    grid_search = GridSearchCV(
        SVC(kernel=kernel, probability=True, random_state=42, class_weight='balanced'),
        param_grid,
        cv=5,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train_scaled, y_train)

    svm = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_cv_score = round(grid_search.best_score_ * 100, 2)

    print(f"\n[Grid Search Results]")
    print(f"  Kernel: {kernel}")
    print(f"  Parameter terbaik: {best_params}")
    print(f"  CV F1-Score terbaik: {best_cv_score}%")

    y_pred = svm.predict(X_test_scaled)
    akurasi = accuracy_score(y_test, y_pred)
    print(f"\n[Test Set Performance]")
    print(f"  Overall Accuracy: {akurasi*100:.2f}%")
    laporan = classification_report(y_test, y_pred,
                                    target_names=KELAS, output_dict=True, digits=4)

    laporan['best_params'] = best_params
    laporan['best_cv_score'] = best_cv_score
    laporan['kernel'] = kernel
    laporan['color_mode'] = color_mode
    laporan['dataset_path'] = dataset_path

    print(f"\n[Per-Class Performance]")
    for kelas in KELAS:
        if kelas in laporan:
            f1 = laporan[kelas]['f1-score']
            recall = laporan[kelas]['recall']
            precision = laporan[kelas]['precision']
            print(f"  {kelas.upper()}:")
            print(f"    - F1-Score: {f1:.4f}")
            print(f"    - Recall (sensitifitas): {recall:.4f}")
            print(f"    - Precision (presisi): {precision:.4f}")

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(svm, f)
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    with open(MODEL_META_PATH, 'w') as f:
        json.dump({'kernel': kernel, 'color_mode': color_mode, 'dataset_path': dataset_path}, f)

    return svm, scaler, akurasi, laporan


def load_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(MODEL_META_PATH):
        with open(MODEL_PATH, 'rb') as f:
            svm = pickle.load(f)
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        with open(MODEL_META_PATH, 'r') as f:
            meta = json.load(f)
        return svm, scaler, meta
    return None, None, None


# ─────────────────────────────────────────────
# PREDIKSI GAMBAR
# ─────────────────────────────────────────────

def prediksi_gambar(image_path):
    svm, scaler, meta = load_model()
    if svm is None:
        return None, 0, []

    color_mode = meta.get('color_mode', 'hsv') if meta else 'hsv'
    fitur = ekstraksi_fitur_glcm(image_path, color_mode=color_mode)
    if fitur is None:
        return None, 0, []

    fitur_scaled = scaler.transform([fitur])
    prediksi = svm.predict(fitur_scaled)[0]
    proba = svm.predict_proba(fitur_scaled)[0]

    label = KELAS[prediksi]
    persentase_kesegaran = hitung_persentase(proba, prediksi)

    detail_proba = [
        {'kelas': KELAS[i], 'prob': round(p * 100, 2)}
        for i, p in enumerate(proba)
    ]

    return label, persentase_kesegaran, detail_proba


def hitung_persentase(proba, prediksi):
    """Konversi probabilitas ke persentase kesegaran (0–100%)"""
    segar_prob  = proba[0]   # index 0 = segar
    layu_prob   = proba[1]   # index 1 = layu
    busuk_prob  = proba[2]   # index 2 = busuk

    # Segar = 100%, Layu = 50%, Busuk = 10%
    persentase = (segar_prob * 100) + (layu_prob * 50) + (busuk_prob * 10)
    return round(persentase, 1)


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.route('/')
def index():
    model_ada = os.path.exists(MODEL_PATH)
    return render_template(
        'index.html',
        model_ada=model_ada,
        dataset_path='',
        selected_kernel='rbf',
        selected_color_mode='hsv',
        kernels=SVM_KERNELS,
        color_modes=COLOR_MODES
    )


@app.route('/info')
def info():
    """Halaman informasi tentang GLCM dan SVM"""
    return render_template('info.html')


@app.route('/train', methods=['POST'])
def train():
    dataset_path = request.form.get('dataset_path', '').strip()
    kernel = request.form.get('kernel', 'rbf')
    color_mode = request.form.get('color_mode', 'hsv')

    if kernel not in SVM_KERNELS:
        kernel = 'rbf'
    if color_mode not in COLOR_MODES:
        color_mode = 'hsv'

    try:
        svm, scaler, akurasi, laporan = training_model(
            dataset_path=dataset_path,
            kernel=kernel,
            color_mode=color_mode
        )
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print('[TRAIN ERROR]', tb)
        return render_template(
            'index.html',
            pesan_error=f"Terjadi error saat training: {e}",
            model_ada=os.path.exists(MODEL_PATH),
            dataset_path=dataset_path,
            selected_kernel=kernel,
            selected_color_mode=color_mode,
            kernels=SVM_KERNELS,
            color_modes=COLOR_MODES
        )

    if svm is None:
        return render_template(
            'index.html',
            pesan_error="Dataset terlalu sedikit, path salah, atau folder kosong!",
            model_ada=False,
            dataset_path=dataset_path,
            selected_kernel=kernel,
            selected_color_mode=color_mode,
            kernels=SVM_KERNELS,
            color_modes=COLOR_MODES
        )

    akurasi_persen = round(akurasi * 100, 2)
    return render_template(
        'index.html',
        model_ada=True,
        pesan_sukses=f"Model berhasil dilatih! Akurasi: {akurasi_persen}%",
        akurasi=akurasi_persen,
        laporan=laporan,
        dataset_path=dataset_path,
        selected_kernel=kernel,
        selected_color_mode=color_mode,
        kernels=SVM_KERNELS,
        color_modes=COLOR_MODES
    )


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return render_template('index.html',
                            pesan_error="File tidak valid! Gunakan PNG/JPG.",
                            model_ada=os.path.exists(MODEL_PATH))

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    label, persentase, detail_proba = prediksi_gambar(filepath)

    if label is None:
        return render_template('index.html',
                            pesan_error="Model belum dilatih! Latih model terlebih dahulu.",
                            model_ada=False)

    return render_template('result.html',
                        label=label,
                        persentase=persentase,
                        detail_proba=detail_proba,
                        image_path=filename)


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    # Disable the reloader on Windows to avoid connection reset issues
    app.run(debug=True, use_reloader=False)
