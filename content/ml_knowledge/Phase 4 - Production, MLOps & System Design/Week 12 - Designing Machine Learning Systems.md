---
title: Week 12 - Designing Machine Learning Systems
phase: Phase 4 - Production, MLOps & System Design
tags: #MLSystemDesign #FeatureStore #DataPipeline #BatchServing #RealtimeServing #ModelRegistry
date: 2026-07-03
---

# 🟢 Week 12 - Designing Machine Learning Systems

> [!NOTE] 📋 ภาพรวมเนื้อหา (Overview)
> **ทฤษฎี (2 ชม.):** การออกแบบสถาปัตยกรรม ML, Batch vs Real-time Serving, Feature Stores  
> **เวิร์กชอป (Workshops):**
> - **Workshop 1:** ออกแบบ Data Pipeline สำหรับ ML System (ETL vs ELT)
> - **Workshop 2:** จำลองระบบ Feature Store เบื้องต้นด้วย Redis / In-memory Dict
> - **Workshop 3:** เปรียบเทียบสถาปัตยกรรม Batch Inference กับ Online Real-time Inference
> - **Workshop 4:** การจัดการ Model Registry ด้วย MLflow เบื้องต้น

---

## 📖 ส่วนที่ 1: สรุปทฤษฎีสำคัญ (Key Theory Concepts)

### 1. ความท้าทายของระบบ ML ในโลกจริง (Production ML Systems)
การเขียนโมเดลใน Notebook เป็นเพียง 5% ของระบบทั้งหมด อีก 95% คือระบบโครงสร้างพื้นฐาน (Infrastructure) เช่น การดึงข้อมูลอัตโนมัติ การตรวจสอบคุณภาพข้อมูล การจัดการเวอร์ชันโมเดล และระบบรองรับผู้ใช้งานหลักล้านคน

### 2. Feature Store & Model Serving
- **Feature Store:** คลังกลางสำหรับจัดเก็บและแชร์ฟีเจอร์ที่คำนวณไว้แล้ว เพื่อให้ทีม Data Science ทั้งบริษัทใช้ร่วมกันได้ ป้องกันปัญหาตรรกะฟีเจอร์ตอน Train กับตอน Serving ไม่ตรงกัน (Training-Serving Skew)
- **Batch Serving:** ทำนายผลล่วงหน้าเป็นรอบ ๆ (เช่น คำนวณโค้ดส่วนลดให้ลูกค้าทุกคนตอนเที่ยงคืน) เก็บผลลง DB
- **Real-time Serving:** ทำนายผลทันทีที่ผู้ใช้กดคลิกในเศษเสี้ยววินาที (เช่น ระบบแนะนำสินค้าตอนไถแอป)

---

## 🛠️ ส่วนที่ 2: เวิร์กชอปเชิงปฏิบัติการ (Hands-on Workshops)

### Workshop 1: ออกแบบ Data Pipeline สำหรับ ML System (ETL vs ELT)
เขียน Python สคริปต์จำลองกระบวนการ Extract, Transform และ Load ข้อมูลเพื่อเตรียมพร้อมสำหรับ Model Training

```python
import pandas as pd
import numpy as np

class MLDataETLPipeline:
    def __init__(self, db_connection_string):
        self.conn_str = db_connection_string

    def extract(self):
        """1. Extract - จำลองการดึงข้อมูลดิบจาก Database หรือ API"""
        print("📥 [Extract] กำลังดึงข้อมูลจากแหล่งต้นทาง...")
        raw_data = pd.DataFrame({
            'user_id': [101, 102, 103, 104],
            'age': [25, 34, np.nan, 45],
            'total_spend': [1500.50, 4200.00, 800.00, 12500.00],
            'last_active_days': [1, 5, 30, 2]
        })
        return raw_data

    def transform(self, df):
        """2. Transform - ทำความสะอาดและวิศวกรรมฟีเจอร์ (Feature Engineering)"""
        print("⚙️ [Transform] ทำความสะอาดและสร้างฟีเจอร์ใหม่...")
        df_clean = df.copy()
        # เติมค่าว่างด้วยค่ามัธยฐาน
        df_clean['age'].fillna(df_clean['age'].median(), inplace=True)
        # สร้างฟีเจอร์ใหม่: ลูกค้าเกรด VIP (ยอดใช้จ่าย > 3000 และเข้าใช้งานล่าสุด < 7 วัน)
        df_clean['is_vip'] = ((df_clean['total_spend'] > 3000) & (df_clean['last_active_days'] < 7)).astype(int)
        return df_clean

    def load(self, df):
        """3. Load - บันทึกข้อมูลที่พร้อมใช้ลง Data Warehouse หรือ Feature Store"""
        print(f"📤 [Load] บันทึกข้อมูลทั้ง {len(df)} แถว ลงสู่ระบบกลางสำเร็จ!")
        return df

# รันระบบ ETL
pipeline = MLDataETLPipeline("mock://db")
raw = pipeline.extract()
processed = pipeline.transform(raw)
pipeline.load(processed)
print("👉 ตัวอย่างข้อมูลหลังผ่าน ETL:\n", processed)
```

---

### Workshop 2: จำลองระบบ Feature Store เบื้องต้น
สร้าง In-memory Feature Store เพื่อแสดงวิธีที่ระบบ Backend หน้าบ้านดึงฟีเจอร์ของลูกค้ามารวมกับโมเดลแบบ Real-time

```python
import time

class MockFeatureStore:
    def __init__(self):
        # จำลอง Redis / Feast Feature Store เก็บค่าฟีเจอร์ล่าสุดของลูกค้าแต่ละคน
        self.store = {
            "user_101": {"age": 25, "total_spend": 1500.50, "is_vip": 0},
            "user_102": {"age": 34, "total_spend": 4200.00, "is_vip": 1},
            "user_104": {"age": 45, "total_spend": 12500.00, "is_vip": 1}
        }

    def get_online_features(self, entity_id: str):
        """ดึงฟีเจอร์แบบ Latency ต่ำมากสำหรับการทำ Online Inference"""
        start_t = time.time()
        features = self.store.get(entity_id, None)
        latency_ms = (time.time() - start_t) * 1000
        print(f"⚡ ดึงฟีเจอร์สำหรับ {entity_id} ใช้เวลาเพียง {latency_ms:.4f} ms")
        return features

# ทดสอบใช้งาน Feature Store ตอน Serving
feature_store = MockFeatureStore()
user_features = feature_store.get_online_features("user_102")
print("ฟีเจอร์ที่ได้นำไปเข้าโมเดลพยากรณ์ต่อ:", user_features)
```

---

### Workshop 3: เปรียบเทียบ Batch Inference กับ Online Real-time Inference
โค้ดจำลองความแตกต่างของการรันทำนายผลแบบชุดใหญ่กลางดึก (Batch) กับแบบเรียลไทม์ทีละคน (Online)

```python
import numpy as np
import time

# จำลองโมเดลพยากรณ์
def mock_predict(features_matrix):
    # สมการจำลอง: โอกาสซื้อสินค้า = sig(spend * 0.001)
    return 1 / (1 + np.exp(-features_matrix[:, 1] * 0.0005))

# 1. Batch Inference (ทำนายล่วงหน้าทีเดียว 100,000 คน)
print("--- 1. จำลอง Batch Inference (รันรอบดึก) ---")
batch_data = np.random.rand(100000, 2) * [60, 10000] # [age, spend]
start_batch = time.time()
batch_preds = mock_predict(batch_data)
print(f"✅ ทำนายลูกค้า 100,000 คน เสร็จสิ้นในเวลา {time.time() - start_batch:.4f} วินาที -> บันทึกผลลง DB เรียบร้อย!")

# 2. Online Real-time Inference (ทำนายทันทีที่ลูกค้ากดเข้าแอป)
print("\n--- 2. จำลอง Online Real-time Inference ---")
single_user_data = np.array([[28, 4500.00]])
start_online = time.time()
single_pred = mock_predict(single_user_data)[0]
print(f"⚡ ทำนายผลลูกค้า 1 ราย เสร็จสิ้นใน {(time.time() - start_online)*1000:.4f} ms -> โอกาสซื้อสินค้า: {single_pred*100:.1f}%")
```

---

### Workshop 4: การจัดการ Model Registry ด้วย MLflow เบื้องต้น
ใช้ไลบรารี **MLflow** เพื่อบันทึกค่าพารามิเตอร์ ผลความแม่นยำ และจัดเก็บตัวไฟล์โมเดลเป็นเวอร์ชันอย่างเป็นระบบ

```python
# หมายเหตุ: หากยังไม่ได้ติดตั้ง ให้รัน -> pip install mlflow scikit-learn
try:
    import mlflow
    import mlflow.sklearn
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    has_mlflow = True
except ImportError:
    has_mlflow = False
    print("กรุณาติดตั้ง mlflow โดยรัน: pip install mlflow")

if has_mlflow:
    # 1. สร้างข้อมูลทดสอบ
    X, y = make_classification(n_samples=500, n_features=10, random_state=42)

    # 2. เริ่มการบันทึกการทดลอง (MLflow Experiment Run)
    mlflow.set_experiment("Promotion_Prediction_Experiment")
    
    with mlflow.start_run(run_name="RandomForest_v1"):
        # กำหนดพารามิเตอร์
        n_est = 100
        max_depth = 5
        
        # บันทึกพารามิเตอร์ลง MLflow
        mlflow.log_param("n_estimators", n_est)
        mlflow.log_param("max_depth", max_depth)
        
        # Train โมเดล
        clf = RandomForestClassifier(n_estimators=n_est, max_depth=max_depth, random_state=42)
        clf.fit(X, y)
        acc = clf.score(X, y)
        
        # บันทึกคะแนนความแม่นยำ (Metric)
        mlflow.log_metric("accuracy", acc)
        
        # บันทึกตัวโมเดลลง Model Registry
        # mlflow.sklearn.log_model(clf, "rf_model")
        
        print("✅ บันทึกการทดลองและเวอร์ชันโมเดลลง MLflow Registry สำเร็จ!")
        print(f" Run Name: RandomForest_v1 | Accuracy: {acc*100:.2f}%")
        print("👉 พิมพ์คำสั่ง 'mlflow ui' ใน Terminal เพื่อเปิดหน้าเว็บจัดการโมเดลดูได้ทันที!")
```

---

## 🔗 อ้างอิงและจุดเชื่อมโยง (Wiki-Links & Next Steps)
- **ภาพรวมเฟส:** [[Phase 4 - Overview|Phase 4: Production, MLOps & System Design]]
- **บทเรียนถัดไป:** [[Week 13 - Model Deployment & Serving]]
- **ความรู้ที่เกี่ยวข้อง:** [[Feature Store Architecture]], [[MLflow Model Registry Guide]], [[Data Pipeline Best Practices]]
