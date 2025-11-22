import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ================================================================
# 1. LOAD & PREPARE DATA
# ================================================================
def load_data():
    # List file sesuai struktur folder yang diupload
    file_paths = [
        'QCM3.csv',
        'QCM6.csv',
        'QCM7.csv',
        'QCM10.csv',
        'QCM12.csv'
    ]

    dfs = []
    for path in file_paths:
        # File menggunakan delimiter ';'
        df = pd.read_csv(path, sep=';')
        dfs.append(df)

    # Gabungkan seluruh file menjadi satu DataFrame
    combined = pd.concat(dfs, ignore_index=True)
    return combined


# Load Data
data = load_data()

# Pisahkan Fitur (X) dan Target (y)
X = data.iloc[:, :10]   # 10 sensor
y_raw = data.iloc[:, 10:]   # 5 kolom target One-Hot

# Konversi One-Hot menjadi 1 label (idxmax)
y = y_raw.idxmax(axis=1)

print("Total sampel data:", len(data))
print("Distribusi Kelas:\n", y.value_counts())

# Split Data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# ================================================================
# 2. IMPLEMENTASI METODE ENSEMBLE
# ================================================================

# A. BAGGING → Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# B. BOOSTING → Gradient Boosting
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)

# C. VOTING (Hard Voting)
voting_clf = VotingClassifier(
    estimators=[('rf', rf_model), ('gb', gb_model)],
    voting='hard'
)
voting_clf.fit(X_train, y_train)
y_pred_vote = voting_clf.predict(X_test)


# ================================================================
# 3. EVALUASI
# ================================================================
print("\n============ HASIL EVALUASI ============\n")

print(f"Akurasi Random Forest (Bagging): {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"Akurasi Gradient Boosting (Boosting): {accuracy_score(y_test, y_pred_gb):.4f}")
print(f"Akurasi Voting Classifier (Max Voting): {accuracy_score(y_test, y_pred_vote):.4f}")

print("\n--- Classification Report (Random Forest) ---")
print(classification_report(y_test, y_pred_rf))

# ================================================================
# 4. Confusion Matrix — Random Forest
# ================================================================
cm = confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel("Prediksi")
plt.ylabel("Aktual")
plt.title("Confusion Matrix - Random Forest")
plt.show()
