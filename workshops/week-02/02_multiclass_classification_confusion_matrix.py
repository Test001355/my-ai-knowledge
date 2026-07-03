# ====================================================================
# Workshop 2: Multiclass & Confusion Matrix
# ====================================================================

# # Workshop 2: การจัดการข้อมูลแบบ Multiclass และสร้าง Confusion Matrix
# ขยายความสามารถสู่การจำแนกข้อมูลที่มีมากกว่า 2 คลาส (Multiclass Classification) เช่น พันธุ์ดอกไม้ หรือ หมวดหมู่สินค้า พร้อมวิเคราะห์จุดอ่อนโมเดลด้วย Confusion Matrix

# ## 1. เตรียมข้อมูลพันธุ์ดอกไม้ Iris (3 คลาส) และฝึกโมเดล

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# โหลดข้อมูล Iris Dataset (Setosa, Versicolor, Virginica)
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# ใช้ Random Forest ในการจำแนก Multiclass
rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

print("--- Multiclass Classification Report ---")
print(classification_report(y_test, y_pred, target_names=target_names))

# ------------------------------------------------------------

# ## 2. สร้างและพล็อต Confusion Matrix เพื่อวิเคราะห์ความถูกต้อง

# คำนวณ Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# พล็อต Heatmap ของ Confusion Matrix
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.title("Confusion Matrix - Iris Flower Classification", fontsize=14)
plt.xlabel("Predicted Label (คำทำนายของโมเดล)", fontsize=12)
plt.ylabel("True Label (ความจริง)", fontsize=12)
plt.tight_layout()
# plt.show() # Uncomment เมื่อรันใน Jupyter Notebook

print("✅ Confusion Matrix Explained:")
print(" - แนวทแยงมุมหลัก (Diagonal) คือจำนวนตัวอย่างที่โมเดลทายถูก 100%")
print(" - ช่องอื่น ๆ คือจุดที่โมเดลเกิดความสับสน (Misclassified)")

# ------------------------------------------------------------
