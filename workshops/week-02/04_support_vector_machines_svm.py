# ====================================================================
# Workshop 4: SVM & Model Comparison
# ====================================================================

# # Workshop 4: การใช้ Support Vector Machines (SVM) และเปรียบเทียบประสิทธิภาพ
# เรียนรู้การทำงานของ Support Vector Machines (SVM) ในการสร้าง Hyperplane แยกคลาสข้อมูล พร้อมเปรียบเทียบประสิทธิภาพระหว่าง Kernel ต่าง ๆ (Linear vs RBF)

# ## 1. การฝึกโมเดล SVM ด้วย Kernel รูปแบบต่าง ๆ

import numpy as np
import pandas as pd
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# สร้างข้อมูลจำลองรูปพระจันทร์เสี้ยว (Non-linear Data) ที่เส้นตรงธรรมดาตัดไม่ขาด
X, y = make_moons(n_samples=300, noise=0.25, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# สเกลข้อมูล
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 1. SVM แบบ Linear Kernel (ใช้เส้นตรงตัด)
svm_linear = SVC(kernel='linear', C=1.0, random_state=42)
svm_linear.fit(X_train, y_train)
pred_linear = svm_linear.predict(X_test)

# 2. SVM แบบ RBF Kernel (Radial Basis Function - เส้นโค้งยืดหยุ่น)
svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_rbf.fit(X_train, y_train)
pred_rbf = svm_rbf.predict(X_test)

print("--- ผลเปรียบเทียบความแม่นยำบน Non-linear Data (Make Moons) ---")
print(f"1. Linear Kernel Accuracy: {accuracy_score(y_test, pred_linear)*100:.2f}% (เส้นตรงตัดข้อมูลโค้งไม่ขาด)")
print(f"2. RBF Kernel Accuracy:    {accuracy_score(y_test, pred_rbf)*100:.2f}% (RBF โค้งรับโครงสร้างข้อมูลได้สมบูรณ์)")

# ------------------------------------------------------------

# ## 2. เปรียบเทียบผลลัพธ์เชิงลึก (Classification Report)

print("\n--- RBF Kernel Classification Report ---")
print(classification_report(y_test, pred_rbf, target_names=['Moon Class 0', 'Moon Class 1']))

print("💡 ข้อแนะนำในการใช้ SVM:")
print(" - สำหรับข้อมูลที่มีมิติสูงมาก (เช่น Text / NLP) Kernel แบบ 'linear' มักจะทำงานได้เร็วและแม่นยำ")
print(" - สำหรับข้อมูลทั่วไปที่มีความซับซ้อน ไม่เป็นเชิงเส้นตรง Kernel แบบ 'rbf' คือตัวเลือกเริ่มต้นที่ดีที่สุด")

# ------------------------------------------------------------
