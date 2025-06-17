#!/usr/bin/env python
# coding: utf-8

# Import Library

# Penjelasan Proses:
# Pada tahap ini, kita mengimpor semua library yang diperlukan untuk proyek machine learning prediksi banjir. Library yang digunakan meliputi:
# 
# - Pandas: untuk manipulasi dan analisis data
# - NumPy: untuk operasi numerik
# - Matplotlib & Seaborn: untuk visualisasi data
# - Scikit-learn: untuk preprocessing, modeling, dan evaluasi
# - XGBoost: untuk algoritma machine learning boosting
# - Scipy: untuk uji statistik

# In[35]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix


# Data Understanding

# Data Loading

# - Memuat dataset banjir dari file CSV ke dalam pandas DataFrame untuk memulai proses analisis data.

# In[36]:


df = pd.read_csv("data_banjir.csv")


# Penjelasan Hasil:
# Dataset berhasil dimuat dan siap untuk tahap eksplorasi data selanjutnya.

# - menampilkan 5 data pertama pada dataset

# In[ ]:


df.head()


# Penjelasan Hasil:
# Dari tampilan awal data, kita dapat melihat struktur dataset dan jenis variabel yang tersedia.

# Analysis Data

# - menampilkan struktur data dan type data setiap kolom pada dataset

# In[ ]:


df.info()


# - menampilkan dan mengecek ada berapa jumlah baris dan kolom pada dataset

# In[ ]:


print(f"Jumlah baris: {df.shape[0]}, Jumlah kolom: {df.shape[1]}")


# - menampilkan dan mengecek jumlah nilai unik pada setiap kolom

# In[ ]:


for col in df.columns:
    print(f"{col}: {df[col].nunique()} nilai unik")


# Penjelasan Hasil:
# Dataset terdiri dari 3000 sampel dengan 15 kolom fitur. Beberapa kolom seperti NAME_3, lat, dan long memiliki kardinalitas tinggi yang perlu dipertimbangkan dalam preprocessing.

# - Menghitung statistik deskriptif untuk fitur numerik guna memahami distribusi dan karakteristik data.

# In[ ]:


df.describe()


# Penjelasan Hasil:
# Statistik deskriptif memberikan gambaran tentang range nilai, mean, median, dan distribusi data numerik yang akan membantu dalam tahap preprocessing.

# - Memeriksa keberadaan missing values dalam dataset untuk menentukan strategi penanganan data yang hilang.

# In[ ]:


print("\nMissing values per kolom:")
print(df.isnull().sum())


# Penjelasan Hasil:
# Insight: Berdasarkan hasil pengecekan, tidak terdapat missing values pada dataset. Hal ini menunjukkan dataset sudah bersih dan siap untuk tahap analisis selanjutnya tanpa memerlukan teknik imputasi data.

# - Menganalisis distribusi variabel target (banjir) untuk memahami keseimbangan kelas dan menentukan apakah diperlukan teknik balancing.

# In[ ]:


banjir_dist = df['banjir'].value_counts()
print("\nDistribusi target (banjir):")
print(banjir_dist)


# In[ ]:


sns.countplot(x='banjir', data=df)
plt.title("Distribusi Label Target: Banjir")
plt.xlabel("Banjir (1 = Ya, 0 = Tidak)")
plt.ylabel("Jumlah Sampel")
plt.show()


# Penjelasan Hasil:
# Insight: Distribusi target menunjukkan keseimbangan yang baik antara kelas positif (banjir) dan negatif (tidak banjir). Hal ini mengindikasikan bahwa tidak diperlukan teknik resampling untuk mengatasi ketidakseimbangan kelas.

# - Mengeksplorasi distribusi fitur kategorikal seperti landcover_class dan wilayah administratif untuk memahami variasi dalam data.

# In[45]:


print("Kategori landcover_class:")
print(df['landcover_class'].value_counts())


# In[46]:


print("\nWilayah NAME_2 (Kabupaten/Kota):")
print(df['NAME_2'].value_counts())


# In[47]:


print("\nWilayah NAME_3 (Kecamatan):")
print(df['NAME_3'].value_counts())


# Penjelasan:
# Insight:
# 
# Fitur kategorikal menunjukkan variasi yang cukup beragam. Kolom NAME_3 (nama kecamatan) memiliki kardinalitas sangat tinggi dengan 1.151 kategori unik, yang dapat menyulitkan model dalam proses pembelajaran. Oleh karena itu, perlu dipertimbangkan untuk menghapus fitur ini atau menggunakan strategi lain dalam preprocessing, seperti encoding khusus (misalnya target encoding atau pengelompokan berdasarkan frekuensi/populasi).

# ### Analisis Keseluruhan Hasil EDA Dataset
# 
# Dataset terdiri dari 3000 sampel dengan 15 kolom fitur. Target `banjir` memiliki distribusi seimbang antara 1 dan 0, sehingga tidak diperlukan teknik penyeimbangan (resampling).
# 
# Beberapa fitur memiliki nilai unik yang tinggi, khususnya `NAME_3` dan `lat/long`, yang dapat menyebabkan sparsity jika diolah tanpa seleksi. Oleh karena itu, fitur-fitur tersebut akan dianalisis lebih lanjut untuk menentukan relevansinya terhadap model. Fitur kategorikal `landcover_class` akan diencoding, sedangkan waktu (`year`, `month`) akan disertakan sebagai fitur prediktif.
# 
# Tidak ditemukan missing value dalam dataset.

# ## Univariate Analysis
# 
# Univariate analysis dilakukan untuk memahami karakteristik masing-masing fitur secara individu. Analisis ini membantu mengidentifikasi distribusi data, pola umum, serta mendeteksi adanya nilai-nilai ekstrim (outlier). Visualisasi juga digunakan untuk mengevaluasi proporsi kategori pada fitur kategorikal.
# 
# UVA dibagi menjadi dua bagian:
# - Analisis fitur numerik: distribusi, outlier
# - Analisis fitur kategorikal: proporsi kelas
# 

# - memilih numerik selain target

# In[ ]:


numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.drop('banjir')


# In[49]:


for col in numerical_cols:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    sns.histplot(df[col], kde=True, ax=axes[0], color='skyblue')
    axes[0].set_title(f'Distribusi Fitur: {col}')
    
    sns.boxplot(x=df[col], ax=axes[1], color='salmon')
    axes[1].set_title(f'Boxplot Fitur: {col}')
    
    plt.tight_layout()
    plt.show()


# - mendeteksi outlier dengan IQR

# In[ ]:


for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outlier_mask = (df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)
    n_outliers = outlier_mask.sum()
    print(f"{col}: {n_outliers} outlier")


# Penjelasan Hasil:
# Insight: Analisis distribusi menunjukkan bahwa beberapa fitur memiliki distribusi yang tidak normal dan terdapat outlier pada beberapa variabel. Informasi ini akan digunakan untuk menentukan strategi preprocessing yang tepat.

# terdapat outlier, tetapi ini tidak dihapus karena outlier tersebut wajar sebab data yang digunakan merupakan data yang bukan angka konsisten seperti harga. disini ada latitude longitude yang memang sangat jauh jaraknya.

# ---------------------------------------------------------------------------------------

# - Menganalisis proporsi dan distribusi fitur kategorikal untuk memahami representasi setiap kategori dalam dataset.

# In[ ]:


categorical_cols = df.select_dtypes(include='object').columns

for col in categorical_cols:
    print(f"\nDistribusi kategori: {col}")
    print(df[col].value_counts(normalize=True).round(2) * 100)

    plt.figure(figsize=(6, 4))
    sns.countplot(y=col, data=df, order=df[col].value_counts().index, palette='pastel')
    plt.title(f'Proporsi Kategori: {col}')
    plt.xlabel("Jumlah Sampel")
    plt.ylabel(col)
    plt.tight_layout()
    plt.show()


# Penjelasan Hasil :
# Insight: Analisis distribusi fitur kategorikal menunjukkan bahwa beberapa fitur seperti NAME_2 dan landcover_class memiliki dominasi kategori tertentu (misal "bogor", "Built-up"), sementara sebagian besar kategori lainnya memiliki representasi yang sangat rendah hingga nol. Ketidakseimbangan ini penting untuk diperhatikan saat feature engineering, karena dapat memengaruhi kinerja model dan menyebabkan bias terhadap kategori mayoritas.

# ## Multivariate Analysis
# 
# Analisis multivariat dilakukan untuk melihat hubungan antar variabel, baik antara fitur numerik satu sama lain maupun antara fitur dengan variabel target (`banjir`). Teknik yang digunakan meliputi:
# 
# - Korelasi Pearson antar fitur numerik
# - Heatmap korelasi
# - Uji Chi-Square untuk fitur kategorikal terhadap label
# 

# - disini menghitung korelasi dan memvisualisasikan heatmap korelasi

# In[ ]:


corr_matrix = df[numerical_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Matriks Korelasi Fitur Numerik")
plt.show()


# #### Penjelasan hasil
# Heatmap korelasi mengungkapkan hubungan linear antar fitur numerik. Korelasi tinggi (>|0.8|) antara fitur seperti avg_rainfall & max_rainfall serta slope & elevation mengindikasikan potensi multikolinearitas, yang perlu diperhatikan terutama dalam model linear. Jika multikolinearitas ditemukan, kita bisa melakukan feature selection atau dimensionality reduction (seperti PCA) untuk mengurangi dampaknya.
# Heatmap korelasi menunjukkan kekuatan dan arah hubungan linear antar fitur numerik. Nilai korelasi berkisar dari:
# 
# +1: korelasi positif sempurna (fitur meningkat bersama),
# 
# 0: tidak ada hubungan linear,
# 
# -1: korelasi negatif sempurna (satu naik, yang lain turun).
# 
# 

# ### Chi-Square Test of Independence
# 
# Untuk fitur kategorikal seperti `landcover_class` dan `NAME_2`, digunakan uji Chi-Square untuk menilai apakah distribusi kategori memiliki hubungan signifikan terhadap label `banjir`.
# 

# - mengambil semua kolom bertipe kategorikal

# In[ ]:


categorical_cols = df.select_dtypes(include='object').columns


# - List untuk menyimpan hasil uji

# In[ ]:


chi_square_results = []


# - Menguji Chi-Square untuk setiap kolom kategorikal terhadap target 'banjir'

# In[ ]:


for col in categorical_cols:
    contingency = pd.crosstab(df[col], df['banjir'])
    chi2, p, dof, expected = chi2_contingency(contingency)

    chi_square_results.append({
        'Fitur': col,
        'Chi-Square': round(chi2, 4),
        'p-value': round(p, 6),
        'Degrees of Freedom': dof,
        'Signifikan (p < 0.05)': 'Ya' if p < 0.05 else 'Tidak'
    })


# - Mengubah ke DataFrame dan menampilkan hasil

# In[ ]:


chi_square_df = pd.DataFrame(chi_square_results)
chi_square_df = chi_square_df.sort_values(by='p-value')

chi_square_df


# - Mengubah nama kolom agar cocok denga DataFrame sebelumnya dan menampilkan hasil rangking fitur dengan plot

# In[ ]:


chi2_df_sorted = chi_square_df.sort_values(by='Chi-Square', ascending=True)

chi2_df_sorted.plot(x='Fitur', y='Chi-Square', kind='barh', figsize=(10, 6), legend=False)
plt.title('Ranking Fitur Berdasarkan Nilai Chi-Square')
plt.xlabel('Chi-Square Value')
plt.tight_layout()
plt.show()


# Penjelasan Hasil:
# Insight: Hasil uji Chi-Square menunjukkan tingkat signifikansi hubungan antara setiap fitur kategorikal dengan target. Fitur dengan p-value < 0.05 dianggap memiliki hubungan yang signifikan dengan kejadian banjir dan akan diprioritaskan dalam model.

# ## Data Preparation

# Tahapan ini mencakup pembersihan data, encoding fitur kategorikal, dan normalisasi fitur numerik sebelum dilakukan pelatihan model.
# 
# - Kolom `landcover_class` dibersihkan dari variasi penulisan dengan lowercasing dan normalisasi label.
# - Kolom `NAME_3`, `lat`, dan `long` dihapus karena memiliki nilai unik sangat banyak (high cardinality).
# - Fitur kategorikal `landcover_class` dan `NAME_2` diencoding dengan metode one-hot.
# - Fitur numerik dinormalisasi menggunakan StandardScaler.
# - Data dibagi ke dalam data latih dan data uji dengan rasio 80:20 secara stratified untuk menjaga proporsi kelas target.

# - Mengcopy dataset asli, agar lebih fleksibel dalam penggunaannya

# In[ ]:


df_prep = df.copy()


# - Memperbaiki inkonsistensi pada landcover_class

# In[ ]:


df_prep['landcover_class'] = df_prep['landcover_class'].str.strip().str.lower()
df_prep['landcover_class'] = df_prep['landcover_class'].replace({
    'built-up': 'built_up',
    'built_up': 'built_up',
    'tree cover': 'tree_cover',
    'tree_cover': 'tree_cover',
    'permanent water bodies': 'water',
    'permanent_water_bodies': 'water',
    'permanent waterbodies': 'water',
})


# - Visualisasi distribusi kategori setelah pembersihan

# In[ ]:


plt.figure(figsize=(8, 4))
sns.countplot(data=df_prep, x='landcover_class', order=df_prep['landcover_class'].value_counts().index)
plt.title("Distribusi landcover_class Setelah Pembersihan")
plt.xlabel("Kategori")
plt.ylabel("Jumlah")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Penjelasan Hasil:
# Insight: Pembersihan data berhasil mengatasi inkonsistensi penulisan pada kategori landcover_class, sehingga kategori yang seharusnya sama tidak lagi terpecah menjadi kategori terpisah.

# Drop kolom

# - Menghapus fitur dengan kardinalitas tinggi (NAME_3, lat, long) yang dapat menyebabkan overfitting dan meningkatkan kompleksitas model tanpa memberikan nilai prediktif yang signifikan.

# In[ ]:


df_prep.drop(columns=['NAME_3', 'lat', 'long'], inplace=True)
print("\nKolom setelah drop kolom high cardinality:")
print(df_prep.columns.tolist())


# Insight: Penghapusan fitur high cardinality mengurangi dimensi data dan kompleksitas model, sambil mempertahankan fitur yang lebih relevan untuk prediksi.

# Encoding Data

# - Mengkonversi fitur kategorikal (landcover_class dan NAME_2) menjadi format numerik menggunakan one-hot encoding agar dapat diproses oleh algoritma machine learning.

# In[ ]:


df_prep = pd.get_dummies(df_prep, columns=['landcover_class', 'NAME_2'], drop_first=True)

print("\nJumlah kolom setelah one-hot encoding:", df_prep.shape[1])
print("Contoh kolom setelah encoding:")
print(df_prep.columns[:15])


# Insight: One-hot encoding berhasil mengkonversi fitur kategorikal menjadi variabel dummy numerik. Jumlah kolom meningkat sesuai dengan jumlah kategori unik dalam setiap fitur kategorikal.

# Splitting Data

# - Membagi dataset menjadi training set dan testing set dengan rasio 80:20. Menggunakan stratified sampling untuk mempertahankan proporsi kelas target yang seimbang.

# In[ ]:


X = df_prep.drop('banjir', axis=1)
y = df_prep['banjir']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# In[63]:


print("\nUkuran data setelah split:")
print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Distribusi target (train):\n{y_train.value_counts()}")
print(f"Distribusi target (test):\n{y_test.value_counts()}")


# Insight: Data berhasil dibagi dengan proporsi yang seimbang. Stratified sampling memastikan bahwa distribusi kelas target pada training dan testing set konsisten dengan dataset asli.

# Normalisasi Data

# - Melakukan normalisasi fitur numerik menggunakan StandardScaler untuk memastikan semua fitur memiliki skala yang sama, yang penting untuk algoritma machine learning yang sensitif terhadap skala data.

# - Menormalisasi fitur numerik (kecuali hasil one-hot)

# In[ ]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# - Mengambil nama kolom numerik (sebelum encoding)

# In[ ]:


numeric_cols_scaled = df.select_dtypes(include=['int64', 'float64']).drop(columns=['banjir', 'lat', 'long']).columns

plt.figure(figsize=(10, 6))
for i, col in enumerate(numeric_cols_scaled[:6]):  # tampilkan 6 pertama
    plt.subplot(2, 3, i+1)
    sns.histplot(X_train_scaled[:, i], bins=30, kde=True, color='skyblue')
    plt.title(f"{col} (scaled)")
plt.tight_layout()
plt.show()


# #### Insight:
# Normalisasi berhasil diterapkan pada fitur numerik. Proses ini dilakukan dengan:
# 
# - Fitting scaler (misalnya StandardScaler) hanya pada data training untuk menghindari data leakage.
# 
# - Kemudian transformasi diterapkan ke data training dan testing menggunakan scaler yang sama.

# Grafik distribusi menunjukkan hasil normalisasi (standardisasi) dari beberapa fitur numerik (fitur dikurangi rata-rata dan dibagi standar deviasi).
# 
# #### Observasi per fitur:
# - avg_rainfall & max_rainfall: Distribusi cenderung positif skewed, namun tetap dalam skala yang seragam (mean mendekati 0).
# 
# - avg_temperature: Distribusi mendekati normal, namun sedikit multimodal.
# 
# - elevation & slope: Terlihat sangat skewed ke kanan, menunjukkan adanya banyak nilai rendah dan beberapa outlier tinggi.
# 
# - ndvi: Sebaran cukup simetris dan mendekati normal, cocok dengan karakteristik NDVI (nilai antara -1 sampai 1 atau lebih tergantung konteks scaling awal).
# 
# 

# ## Modeling
# 
# Pada tahap ini, dua algoritma klasifikasi digunakan:
# 
# 1. **Random Forest**: model berbasis ensemble decision tree yang tangguh terhadap overfitting dan mampu menangani data numerik dan kategorikal.
# 2. **XGBoost**: model boosting yang populer karena efisiensi dan akurasinya.
# 
# Evaluasi dilakukan menggunakan metrik: accuracy, precision, recall, dan F1-score pada data uji.

# - Melatih model Random Forest Classifier, yang merupakan algoritma ensemble berbasis decision tree yang robust terhadap overfitting dan mampu menangani fitur numerik dan kategorikal.

# In[ ]:


rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_scaled, y_train)


# In[67]:


print("=== Random Forest ===")
y_pred_rf = rf_model.predict(X_test_scaled)
print(classification_report(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))


# - Melatih model XGBoost Classifier, yang merupakan algoritma boosting yang populer karena efisiensi dan akurasi tinggi dalam berbagai kompetisi machine learning.

# In[ ]:


xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train_scaled, y_train)

print("\n=== XGBoost ===")
y_pred_xgb = xgb_model.predict(X_test_scaled)
print(classification_report(y_test, y_pred_xgb))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_xgb))


# - Membandingkan performa kedua model berdasarkan akurasi untuk menentukan model terbaik untuk prediksi banjir.

# In[69]:


rf_acc = rf_model.score(X_test_scaled, y_test)
xgb_acc = xgb_model.score(X_test_scaled, y_test)

print(f"Akurasi Random Forest: {rf_acc:.4f}")
print(f"Akurasi XGBoost:       {xgb_acc:.4f}")


# #### Insight Evaluasi Model:
# Perbandingan performa antara model Random Forest dan XGBoost menunjukkan bahwa keduanya memberikan hasil yang sangat baik dalam klasifikasi kejadian banjir, dengan akurasi masing-masing:
# 
# - Random Forest: 93.67%
# 
# - XGBoost: 93.17%
# 
# Namun, jika dilihat lebih dalam melalui precision dan recall:
# 
# - Random Forest lebih unggul dalam recall untuk kelas 0 (tidak banjir) (1.00), artinya sangat baik dalam mengenali wilayah yang benar-benar tidak terkena banjir.
# 
# - XGBoost lebih baik dalam precision untuk kelas 1 (banjir) (0.96), artinya lebih minim false positive dalam memprediksi wilayah banjir.
# 
# Confusion matrix juga mendukung hal ini:
# 
# - Random Forest salah mengklasifikasikan 37 wilayah banjir sebagai tidak banjir.
# 
# - XGBoost hanya salah 31 pada kasus tersebut, namun lebih sering salah mengklasifikasikan wilayah tidak banjir sebagai banjir.
# 
# Oleh karena itu, pemilihan model akhir bisa didasarkan pada prioritas bisnis:
# 
# - Jika lebih penting untuk tidak melewatkan prediksi banjir (recall tinggi), maka Random Forest lebih cocok.
# 
# - Jika lebih penting untuk menghindari false alarm (precision tinggi), maka XGBoost bisa dipertimbangkan.

# #### Kesimpulan Akhir:
# Proyek machine learning untuk prediksi banjir telah berhasil diselesaikan melalui tahapan yang terstruktur dan komprehensif, yaitu:
# 
# 1. Data Understanding:
# Dataset terdiri dari 3000 sampel dan 15 fitur numerik dan kategorikal yang telah dipastikan bebas dari nilai hilang.
# 
# 2. Exploratory Data Analysis (EDA):
# Dilakukan analisis statistik dan visualisasi korelasi serta distribusi fitur untuk memahami pola-pola penting dalam data.
# 
# 3. Data Preparation:
# Meliputi pembersihan data, encoding fitur kategorikal, normalisasi fitur numerik, dan pembagian data training-testing secara tepat.
# 
# 4. Modeling:
# Dua model andal, Random Forest dan XGBoost, diterapkan untuk membangun sistem prediksi banjir.
# 
# 5. Evaluation:
# Performa model dibandingkan menggunakan metrik akurasi, precision, recall, f1-score, dan confusion matrix.
# Kedua model menunjukkan hasil yang baik, dengan akurasi di atas 93%, menunjukkan potensi tinggi dalam pemanfaatan model ini untuk mitigasi risiko banjir.
