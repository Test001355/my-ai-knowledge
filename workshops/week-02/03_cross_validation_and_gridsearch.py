# ====================================================================
# Workshop 3: Cross-Validation & GridSearch
# ====================================================================

# # Workshop 3: การทำ Cross-validation และ Hyperparameter Tuning (GridSearch)
# เรียนรู้วิธีการตรวจสอบความเก่งที่แท้จริงของโมเดลด้วย k-Fold Cross Validation และการค้นหาพารามิเตอร์ที่ดีที่สุดอัตโนมัติด้วย GridSearchCV

# ## 1. การทำ k-Fold Cross Validation เพื่อป้องกัน Overfitting

import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

# โหลดข้อมูลไวน์ (Wine Dataset - จำแนกไวน์ 3 ประเภท)
wine = load_wine()
X, y = wine.data, wine.target

# สร้างโมเดล k-Nearest Neighbors (kNN) เบื้องต้น
knn = KNeighborsClassifier(n_neighbors=5)

# ทำ 5-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(knn, X, y, cv=kf, scoring='accuracy')

print("--- 5-Fold Cross Validation Results ---")
print("Scores ในแต่ละรอบ:", [f"{s:.4f}" for s in cv_scores])
print(f"ค่าเฉลี่ยความแม่นยำ (Mean CV Accuracy): {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

# ------------------------------------------------------------

# ## 2. การจูนพารามิเตอร์อัตโนมัติด้วย GridSearchCV

# กำหนดตารางค้นหาพารามิเตอร์ (Parameter Grid)
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11, 15],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

# ตั้งค่า GridSearchCV
grid_search = GridSearchCV(
    estimator=KNeighborsClassifier(),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1 # ใช้ CPU ทุกคอร์ในการคำนวณ
)

# เริ่มการค้นหา (Train & Search)
print("⏳ กำลังค้นหา Hyperparameter ที่ดีที่สุดด้วย GridSearchCV...")
grid_search.fit(X, y)

print("\n--- ผลลัพธ์การค้นหา Hyperparameter ---")
print("พารามิเตอร์ที่ดีที่สุด (Best Parameters):", grid_search.best_params_)
print(f"ความแม่นยำสูงสุดที่ทำได้ (Best CV Score): {grid_search.best_score_:.4f} ({grid_search.best_score_*100:.2f}%)")

# ------------------------------------------------------------
