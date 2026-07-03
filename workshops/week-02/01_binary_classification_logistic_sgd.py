# ====================================================================
# Workshop 1: Binary Classification
# ====================================================================

# # Workshop 1: การสร้างโมเดล Binary Classification (Logistic Regression / SGD)
# เรียนรู้การสร้างโมเดลเพื่อจำแนกประเภทข้อมูล 2 กลุ่ม (Binary Classification) เช่น การตรวจจับผู้ป่วยเบาหวาน พร้อมวัดผลด้วย Precision, Recall และ ROC Curve

# ## 1. เตรียมชุดข้อมูลและการทำ Feature Scaling

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# โหลดชุดข้อมูลมะเร็งเต้านม (0 = Malignant/เป็นโรค, 1 = Benign/ไม่เป็นโรค)
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# แบ่ง Train 80% / Test 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature Scaling (จำเป็นมากสำหรับ Logistic Regression และ SGD)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"จำนวนข้อมูลทั้งหมด: {len(X)} | Train: {len(X_train)} | Test: {len(X_test)}")
print("ตัวอย่างคอลัมน์ฟีเจอร์:", list(X.columns[:5]))

# ------------------------------------------------------------

# ## 2. ฝึกสอนโมเดล Logistic Regression และ SGD Classifier

# 1. Logistic Regression
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train_scaled, y_train)
y_pred_log = log_reg.predict(X_test_scaled)

# 2. Stochastic Gradient Descent (SGD Classifier)
sgd_clf = SGDClassifier(loss='log_loss', max_iter=1000, random_state=42)
sgd_clf.fit(X_train_scaled, y_train)
y_pred_sgd = sgd_clf.predict(X_test_scaled)

def evaluate_binary(y_true, y_pred, model_name):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"--- {model_name} Performance ---")
    print(f"Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"Precision: {prec:.4f} (ทำนายว่าเป็นจริง ถูกต้องกี่ %)")
    print(f"Recall:    {rec:.4f} (จับความเป็นจริงได้ครบกี่ %)")
    print(f"F1-Score:  {f1:.4f}\n")

evaluate_binary(y_test, y_pred_log, "Logistic Regression")
evaluate_binary(y_test, y_pred_sgd, "SGD Classifier")

# ------------------------------------------------------------
