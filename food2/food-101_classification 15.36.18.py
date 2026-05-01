# =========================================
# FOOD-101 SINIFLANDIRICI — DATA LEAKAGE DÜZELTİLMİŞ
# M4 MacBook Pro (10 CPU, 16 GB RAM) için optimize edilmiştir
#
# VERİ AKIŞI:
#   Tüm veri → %70 train | %15 validation | %15 test
#   - train      : model eğitimi
#   - validation : AutoML model seçimi (hiperparametre)
#   - test       : SADECE final değerlendirme (model bunu hiç görmez)
# =========================================

# =========================================
# 1️⃣ KÜTÜPHANELER
# =========================================

import os
import math
import numpy as np
import tensorflow as tf
import kagglehub
from tqdm import tqdm
import pandas as pd
import h2o
from h2o.automl import H2OAutoML

print("=" * 60)
print("Tensorflow:", tf.__version__)
print("Fiziksel cihazlar:", tf.config.list_physical_devices())

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("Metal GPU bulundu:", gpus)
else:
    print("GPU bulunamadı, CPU kullanılacak (M4 için normaldir).")
print("=" * 60)

# =========================================
# 2️⃣ DATASET İNDİR
# =========================================

print("\nDataset indiriliyor...")
path = kagglehub.dataset_download("dansbecker/food-101")

possible_dirs = [
    os.path.join(path, "images"),
    os.path.join(path, "food-101", "images"),
    os.path.join(path, "food-101", "food-101", "images"),
]

images_dir = None
for d in possible_dirs:
    if os.path.exists(d):
        images_dir = d
        break

if images_dir is None:
    raise FileNotFoundError(
        f"'images' klasörü bulunamadı.\n"
        f"İndirilen path içeriği: {os.listdir(path)}"
    )

print("Dataset dizini:", images_dir)
print("Toplam sınıf klasörü:", len(os.listdir(images_dir)))

# =========================================
# 3️⃣ DATASET YÜKLE VE BÖLE (70 / 15 / 15)
#
# Leakage'ı önlemek için veriyi üçe bölüyoruz:
#   - train      (%70): model eğitimi
#   - validation (%15): AutoML model seçimi
#   - test       (%15): yalnızca final değerlendirme
#
# Adımlar:
#   1. image_dataset_from_directory ile %70 train, %30 temp alıyoruz
#   2. temp'i (shuffle edilmiş) yarı yarıya bölüyoruz → %15 val, %15 test
# =========================================

IMG_SIZE   = (224, 224)
BATCH_SIZE = 16
SEED       = 42

print("\nDataset yükleniyor...")

# --- %70 Train ---
train_ds = tf.keras.utils.image_dataset_from_directory(
    images_dir,
    validation_split=0.30,   # %30'u dışarıda bırak
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# --- %30 Temp (val + test için) ---
temp_ds = tf.keras.utils.image_dataset_from_directory(
    images_dir,
    validation_split=0.30,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=1,            # batch=1 → tam sayıda bölme için
    shuffle=False            # bölme tutarlılığı için shuffle kapalı
)

class_names = train_ds.class_names
print("Toplam sınıf sayısı:", len(class_names))

# Temp'i ikiye böl → %15 val, %15 test
temp_size = tf.data.experimental.cardinality(temp_ds).numpy()
val_size  = temp_size // 2

print(f"Temp dataset boyutu : {temp_size} örnek")
print(f"Validation boyutu   : {val_size} örnek")
print(f"Test boyutu         : {temp_size - val_size} örnek")

val_ds_unbatched  = temp_ds.take(val_size)
test_ds_unbatched = temp_ds.skip(val_size)

# Yeniden batch'le
val_ds  = val_ds_unbatched.rebatch(BATCH_SIZE)
test_ds = test_ds_unbatched.rebatch(BATCH_SIZE)

# Pipeline hızlandırma
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds   = val_ds.prefetch(buffer_size=AUTOTUNE)
test_ds  = test_ds.prefetch(buffer_size=AUTOTUNE)

# =========================================
# 4️⃣ FEATURE EXTRACTOR MODEL
# =========================================

print("\nEfficientNetB0 ImageNet ağırlıkları yükleniyor...")

base_model = tf.keras.applications.EfficientNetB0(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet",
    pooling="avg"
)
base_model.trainable = False  # ağırlıklar donduruldu — sadece feature extractor

print("Model hazır! Parametre sayısı:", base_model.count_params())

# =========================================
# 5️⃣ FEATURE KAYIT KLASÖRÜ
# =========================================

os.makedirs("features", exist_ok=True)

# =========================================
# 6️⃣ FEATURE EXTRACTION FONKSİYONU
# =========================================

def extract_and_save(dataset, name):
    """
    Dataset'ten EfficientNetB0 feature'larını çıkarır ve .npy olarak kaydeder.
    Dosya zaten varsa yeniden hesaplamaz (cache).
    """
    feature_file = f"features/{name}_features.npy"
    label_file   = f"features/{name}_labels.npy"

    if os.path.exists(feature_file) and os.path.exists(label_file):
        print(f"\n[CACHE] {name} feature dosyaları mevcut, yükleniyor...")
        features = np.load(feature_file)
        labels   = np.load(label_file)
        print(f"Feature shape: {features.shape} | Label shape: {labels.shape}")
        return features, labels

    features_list = []
    labels_list   = []

    total_batches = tf.data.experimental.cardinality(dataset).numpy()
    if total_batches < 0:
        total_batches = None

    print(f"\n[EXTRACTION] {name} başlıyor... Toplam batch: {total_batches}")

    for images, label in tqdm(dataset, total=total_batches, desc=name):
        images_preprocessed = tf.keras.applications.efficientnet.preprocess_input(images)
        batch_features = base_model(images_preprocessed, training=False).numpy()
        features_list.append(batch_features)
        labels_list.append(label.numpy())

    features = np.vstack(features_list)
    labels   = np.concatenate(labels_list)

    np.save(feature_file, features)
    np.save(label_file,   labels)

    print(f"[KAYIT] {name} → {feature_file}")
    print(f"Feature shape: {features.shape} | Label shape: {labels.shape}")
    return features, labels

# =========================================
# 7️⃣ TRAIN / VALIDATION / TEST EXTRACTION
# =========================================

X_train, y_train = extract_and_save(train_ds, "train")
X_val,   y_val   = extract_and_save(val_ds,   "validation")
X_test,  y_test  = extract_and_save(test_ds,  "test")

print("\nFeature extraction tamamlandı!")
print(f"Train      : {X_train.shape}")
print(f"Validation : {X_val.shape}")
print(f"Test       : {X_test.shape}")

# =========================================
# 8️⃣ H2O BAŞLAT
# =========================================

print("\nH2O başlatılıyor...")

try:
    h2o.cluster().shutdown(prompt=False)
except Exception:
    pass

h2o.init(
    max_mem_size="6G",
    nthreads=-1,
    port=54321
)

print("H2O versiyon:", h2o.__version__)

# XGBoost M4 (Apple Silicon) üzerinde H2O ile desteklenmez
xgb_available = False
try:
    from h2o.estimators.xgboost import H2OXGBoostEstimator
    xgb_available = H2OXGBoostEstimator.available()
except Exception:
    pass

print(f"XGBoost kullanılabilir: {xgb_available}")
if not xgb_available:
    print("⚠️  M4 Apple Silicon'da H2O XGBoost desteklenmez.")
    print("    AutoML → GBM, GLM, DeepLearning, StackedEnsemble kullanacak.")

# =========================================
# 9️⃣ DATAFRAME & H2O FRAME OLUŞTUR
# =========================================

print("\nH2OFrame oluşturuluyor...")

feature_cols = [f"f{i}" for i in range(X_train.shape[1])]  # f0 ... f1279

train_df = pd.DataFrame(X_train, columns=feature_cols)
train_df["label"] = y_train.astype(int)

val_df = pd.DataFrame(X_val, columns=feature_cols)
val_df["label"] = y_val.astype(int)

test_df = pd.DataFrame(X_test, columns=feature_cols)
test_df["label"] = y_test.astype(int)

train_h2o = h2o.H2OFrame(train_df)
val_h2o   = h2o.H2OFrame(val_df)
test_h2o  = h2o.H2OFrame(test_df)   # ← AutoML bunu hiç görmeyecek

# Label'ı kategorik yap
train_h2o["label"] = train_h2o["label"].asfactor()
val_h2o["label"]   = val_h2o["label"].asfactor()
test_h2o["label"]  = test_h2o["label"].asfactor()

x_cols = feature_cols
y_col  = "label"

print(f"Train H2OFrame      : {train_h2o.shape}")
print(f"Validation H2OFrame : {val_h2o.shape}")
print(f"Test H2OFrame       : {test_h2o.shape}  ← model bunu görmeyecek")

# =========================================
# 🔟 AUTOML EĞİTİMİ
#
# LEAKAGE ÖNLEME:
#   - AutoML sadece train_h2o + val_h2o görür
#   - nfolds=0 → validation_frame kullanıldığında CV çakışmasını önler
#   - test_h2o SADECE aşağıdaki final değerlendirmede kullanılır
# =========================================

MAX_RUNTIME = 3600  # saniye — 1 saat (isteğe göre değiştir)

print(f"\nAutoML başlıyor... (max süre: {MAX_RUNTIME // 60} dakika)")

aml = H2OAutoML(
    max_models=20,
    max_runtime_secs=MAX_RUNTIME,
    seed=SEED,
    nfolds=0,                    # validation_frame verildiğinde CV kapatılmalı
                                 # aksi hâlde H2O ikisini birden kullanır → leakage riski
    sort_metric="mean_per_class_error",
    exclude_algos=[] if xgb_available else ["XGBoost"],
    verbosity="info"
)

aml.train(
    x=x_cols,
    y=y_col,
    training_frame=train_h2o,
    validation_frame=val_h2o    # sadece model seçimi için kullanılır
)

# =========================================
# 1️⃣1️⃣ LEADERBOARD
# =========================================

print("\n" + "=" * 60)
print("LEADERBOARD (validation setine göre sıralı):")
print("=" * 60)
lb = aml.leaderboard
print(lb.head(rows=10))

best_model = aml.leader
print(f"\nEn iyi model: {best_model.model_id}")

# =========================================
# 1️⃣2️⃣ FINAL DEĞERLENDİRME — SADECE TEST SETİ
# =========================================
import os
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score as f1_func

print("\n" + "=" * 60)
print("FINAL PERFORMANS — TEST SETİ (model bunu hiç görmedi):")
print("=" * 60)

perf_test = best_model.model_performance(test_h2o)
pred_h2o = best_model.predict(test_h2o)

pred_df  = pred_h2o.as_data_frame()
true_df  = test_h2o["label"].as_data_frame()

# Gerçek ve Tahmin edilen etiketleri numpy dizisine alıyoruz
y_true = true_df["label"].astype(str).values
y_pred = pred_df["predict"].astype(str).values

# --- METRİK HESAPLAMALARI ---
# 1. Top-1 Accuracy
accuracy = np.mean(y_pred == y_true)

# 2. Precision, Recall ve F1-Score (Macro average ile sınıflar arası adil dağılım)
precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
f1_score = f1_func(y_true, y_pred, average='macro', zero_division=0)

# 3. Top-5 Accuracy Hesaplaması
# 'predict' sütununu atıp sadece sınıfların olasılık değerlerini (p0, p1... p100) bırakıyoruz
prob_df = pred_df.drop("predict", axis=1)
class_names = prob_df.columns.values
probs = prob_df.values

# Her satır için en yüksek olasılığa sahip 5 sınıfın indeksini alıyoruz
top5_indices = np.argsort(probs, axis=1)[:, -5:]

top5_acc_count = 0
for i in range(len(y_true)):
    # Gerçek etiket, tahmin edilen en iyi 5 sınıfın içinde mi?
    if y_true[i] in class_names[top5_indices[i]]:
        top5_acc_count += 1
top5_accuracy = top5_acc_count / len(y_true)

print(f"Test Logloss            : {perf_test.logloss():.4f}")
print(f"Test Mean Per Class Err : {perf_test.mean_per_class_error():.4f}")

# Validation performansı ile karşılaştır (Overfitting Kontrolü)
perf_val = best_model.model_performance(val_h2o)
val_acc  = 1.0 - perf_val.mean_per_class_error()

print(f"\nValidation Accuracy     : {val_acc:.4f}  ({val_acc * 100:.2f}%)")
print(f"Fark (val - test)       : {(val_acc - accuracy) * 100:+.2f}%")
if abs(val_acc - accuracy) > 0.05:
    print("⚠️  Fark >5% — hafif overfitting olabilir.")
else:
    print("✅  Fark makul seviyede — model iyi genelliyor.")



# =========================================
# 1️⃣3️⃣ MODELİ KAYDET
# =========================================

save_dir = "./best_food_model"
os.makedirs(save_dir, exist_ok=True)

model_path = h2o.save_model(
    model=best_model,
    path=save_dir,
    force=True
)
print(f"\nModel kaydedildi: {model_path}")

# =========================================
# 1️⃣4️⃣ TEMİZLİK VE KAPANIŞ
# =========================================

h2o.cluster().shutdown(prompt=False)
print("\nH2O kapatıldı.\n")

# Görseldeki tasarıma birebir uygun Performans Karnesi
print("=" * 60)
print("🚀 MODELİN DETAYLI PERFORMANS KARNESİ")
print("=" * 60)
print(f"Top-1 Başarı Oranı (Accuracy): %{accuracy * 100:.2f}")
print(f"Top-5 Başarı Oranı (Esnek):    %{top5_accuracy * 100:.2f}")
print("-" * 60)
print(f"Precision (Kesinlik):          %{precision * 100:.2f}")
print(f"Recall (Duyarlılık):           %{recall * 100:.2f}")
print(f"F1-Score (Genel Skor):         %{f1_score * 100:.2f}")
print("=" * 60)

# Kapanış mesajı ve model yolu
print(f"\n✅ Tüm işlemler başarıyla tamamlandı!")
print(f"💾 Model Kayıt Yolu: {model_path}")
print("=" * 60)
