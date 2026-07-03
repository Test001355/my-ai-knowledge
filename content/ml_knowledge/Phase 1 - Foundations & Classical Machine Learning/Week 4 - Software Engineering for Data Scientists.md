---
title: Week 4 - Software Engineering for Data Scientists
phase: Phase 1 - Foundations & Classical Machine Learning
tags: #SoftwareEngineering #OOP #FastAPI #Logging #Multiprocessing #MemoryOptimization #CleanCode
date: 2026-07-03
---

# 🟢 Week 4 - Software Engineering for Data Scientists

> [!NOTE] 📋 ภาพรวมเนื้อหา (Overview)
> **ทฤษฎี (2 ชม.):** Object-Oriented Programming (OOP) สำหรับ Data Science, การเขียนโค้ดให้ทำงานเร็วขึ้นและใช้หน่วยความจำน้อยลง  
> **เวิร์กชอป (Workshops):**
> - **Workshop 1:** การเขียน Python แบบ OOP (สร้าง Class สำหรับ Data Pipeline)
> - **Workshop 2:** Exception Handling และ Logging ในระบบ ML
> - **Workshop 3:** การจัดการ Memory (Memory profiling) และการทำ Multiprocessing เบื้องต้น
> - **Workshop 4:** สร้าง RESTful API พื้นฐานด้วย FastAPI เพื่อรองรับ Model Inference

---

## 📖 ส่วนที่ 1: สรุปทฤษฎีสำคัญ (Key Theory Concepts)

### 1. ทำไม Data Scientist ต้องรู้ Software Engineering?
โค้ด ML ใน Jupyter Notebook ส่วนใหญ่มักเป็นสปาเก็ตตี (Spaghetti Code) รันได้แค่ครั้งเดียวบนเครื่องเรา แต่นำไปใช้งานจริงในระบบ Production ไม่ได้ การนำหลักการ Software Engineering มาใช้จะช่วยให้:
- **Modularity:** แบ่งโค้ดเป็นส่วนย่อย ๆ (Class / Module) แก้ไขง่าย ไม่กระทบส่วนอื่น
- **Reproducibility:** คนอื่นในทีมสามารถนำไปรันต่อได้ผลลัพธ์เดิมเสมอ
- **Scalability:** รองรับข้อมูลปริมาณมหาศาล (Big Data) และผู้ใช้งานหลายพันคนพร้อมกัน

### 2. หลักการ OOP (Object-Oriented Programming) สำหรับ ML
- **Encapsulation:** ซ่อนความซับซ้อนของอัลกอริทึมไว้ใต้ Class เดียว เช่น เรียกใช้แค่ `pipeline.fit_transform(data)`
- **Inheritance & Polymorphism:** สร้าง Class แม่สำหรับโมเดลมาตรฐาน แล้วสร้าง Class ลูกมาสืบทอดคุณสมบัติเพื่อปรับแต่งเฉพาะงาน

```mermaid
graph TD
    A[BaseMLPipeline<br>- load_data()<br>- evaluate()] --> B[ClassificationPipeline<br>- train_classifier()]
    A --> C[RegressionPipeline<br>- train_regressor()]
```

---

## 🛠️ ส่วนที่ 2: เวิร์กชอปเชิงปฏิบัติการ (Hands-on Workshops)

### Workshop 1: การเขียน Python แบบ OOP (สร้าง Class สำหรับ Data Pipeline)
เปลี่ยนโค้ดสปาเก็ตตีให้กลายเป็น Class `MLDataPipeline` ที่สามารถนำกลับมาใช้ซ้ำได้ (Reusability)

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class MLDataPipeline:
    def __init__(self, target_column, test_size=0.2, random_state=42):
        """กำหนดคุณสมบัติเริ่มต้นของ Pipeline"""
        self.target_col = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.feature_names = None

    def clean_data(self, df):
        """ทำความสะอาดข้อมูล: เติมค่าว่างและลบแถวที่ซ้ำ"""
        df_cleaned = df.drop_duplicates().copy()
        # เติมค่าว่างที่เป็นตัวเลขด้วยค่าเฉลี่ย
        num_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            df_cleaned[col].fillna(df_cleaned[col].mean(), inplace=True)
        return df_cleaned

    def split_and_scale(self, df):
        """แยกฟีเจอร์/เลเบล แบ่ง Train/Test และปรับสเกล"""
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]
        self.feature_names = X.columns.tolist()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        # Fit สเกลเฉพาะ X_train เพื่อป้องกัน Data Leakage
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test

# --- ทดสอบเรียกใช้งาน Data Pipeline ---
if __name__ == "__main__":
    # จำลองข้อมูล
    raw_data = pd.DataFrame({
        'Age': [25, 30, np.nan, 45, 50, 50],
        'Salary': [50000, 60000, 75000, 80000, 90000, 90000],
        'Purchased': [0, 0, 1, 1, 1, 1]
    })

    pipeline = MLDataPipeline(target_column='Purchased')
    cleaned_df = pipeline.clean_data(raw_data)
    X_tr, X_te, y_tr, y_te = pipeline.split_and_scale(cleaned_df)

    print("--- ผลลัพธ์จาก OOP Data Pipeline ---")
    print(f"Train features shape: {X_tr.shape}")
    print(f"Features list: {pipeline.feature_names}")
```

---

### Workshop 2: Exception Handling และ Logging ในระบบ ML
แทนที่ `print()` ด้วยระบบ `logging` มาตรฐาน และดักจับข้อผิดพลาด (Exception) ไม่ให้โปรแกรมแฮงค์เมื่อเจอข้อมูลผิดปกติ

```python
import logging
import sys

# 1. ตั้งค่าระบบ Logging เพื่อบันทึกการทำงานและข้อผิดพลาด
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout), # แสดงผลออกหน้าจอ
        # logging.FileHandler("ml_pipeline.log") # (ตัวเลือก) บันทึกลงไฟล์
    ]
)

logger = logging.getLogger("ML_Logger")

def predict_salary(age, experience):
    """ฟังก์ชันจำลองการทำนายที่มีการดักจับ Exception"""
    logger.info(f"เริ่มการทำนายสำหรับ Age={age}, Exp={experience}")
    try:
        if experience < 0 or age < 18:
            raise ValueError("อายุต้องไม่ต่ำกว่า 18 และประสบการณ์ต้องไม่ติดลบ!")
        
        # สมการจำลอง: เงินเดือน = 30000 + (ประสบการณ์ * 5000)
        predicted_salary = 30000 + (experience * 5000)
        logger.info(f"ทำนายผลสำเร็จ: {predicted_salary:,.2f} บาท")
        return predicted_salary

    except ValueError as ve:
        logger.warning(f"ข้อมูลนำเข้าไม่ถูกต้อง: {ve}")
        return None
    except Exception as e:
        logger.error(f"เกิดข้อผิดพลาดร้ายแรงในระบบ: {str(e)}", exc_info=True)
        return None

# --- ทดสอบการทำงาน ---
print("\n--- ทดสอบระบบ Logging & Exception Handling ---")
predict_salary(age=25, experience=3)  # กรณีปกติ (Success)
predict_salary(age=15, experience=1)  # กรณีข้อมูลผิด (Warning)
predict_salary(age=30, experience=-2) # กรณีค่าติดลบ (Warning)
```

---

### Workshop 3: การจัดการ Memory และการทำ Multiprocessing เบื้องต้น
เทคนิคการจัดการหน่วยความจำสำหรับ Big Data ด้วยการแปลง Data Types และใช้ Python `multiprocessing` ประมวลผลพร้อมกันหลายคอร์

```python
import numpy as np
import pandas as pd
import time
from multiprocessing import Pool, cpu_count

# ----------------------------------------------------
# 1. เทคนิคลดการใช้ Memory (Memory Optimization)
# ----------------------------------------------------
# สร้างตารางขนาดใหญ่จำลอง (100,000 แถว)
large_df = pd.DataFrame({
    'id': np.arange(100000),
    'age': np.random.randint(18, 70, size=100000),
    'score': np.random.rand(100000)
})

mem_before = large_df.memory_usage(index=True).sum() / 1024**2
print(f"Memory ก่อนปรับ: {mem_before:.4f} MB")

# ปรับเปลี่ยน Type จาก int64/float64 เป็นตัวเล็กอย่าง int32/float32
large_df['id'] = large_df['id'].astype(np.int32)
large_df['age'] = large_df['age'].astype(np.int8)  # อายุใช้แค่ int8 (-128 ถึง 127 ก็พอ)
large_df['score'] = large_df['score'].astype(np.float32)

mem_after = large_df.memory_usage(index=True).sum() / 1024**2
print(f"Memory หลังปรับ:  {mem_after:.4f} MB (ประหยัดลง {(mem_before-mem_after)/mem_before*100:.1f}%)")
print("-" * 50)

# ----------------------------------------------------
# 2. การทำ Multiprocessing ประมวลผลแบบขนาน (Parallel)
# ----------------------------------------------------
def heavy_computation(num):
    """จำลองงานคำนวณหนักๆ เช่น การแปลงฟีเจอร์หรือคำนวณทางคณิตศาสตร์"""
    return sum(i * i for i in range(num))

if __name__ == "__main__":
    numbers = [500000, 600000, 700000, 800000] * 2
    cores = cpu_count()
    print(f"จำนวน CPU Cores ที่ใช้ได้ในเครื่องคุณ: {cores} Cores")

    # 1. รันแบบปกติ (Sequential / คอร์เดียว)
    start_time = time.time()
    results_seq = [heavy_computation(n) for n in numbers]
    seq_time = time.time() - start_time
    print(f"เวลาที่ใช้ (Sequential): {seq_time:.4f} วินาที")

    # 2. รันแบบขนาน (Parallel / ใช้ทุกคอร์ช่วยกันคิด)
    start_time = time.time()
    with Pool(processes=min(4, cores)) as pool:
        results_par = pool.map(heavy_computation, numbers)
    par_time = time.time() - start_time
    print(f"เวลาที่ใช้ (Parallel):   {par_time:.4f} วินาที (เร็วขึ้น {seq_time/par_time:.1f} เท่า!)")
```

---

### Workshop 4: สร้าง RESTful API พื้นฐานด้วย FastAPI เพื่อรองรับ Model Inference
เปลี่ยนโมเดล ML ให้กลายเป็น Web API ที่ Software Engineer หน้าบ้าน (Frontend / Mobile) สามารถยิง HTTP Request เข้ามาขอผลคำทำนายได้ทันที

```python
# หมายเหตุ: หากยังไม่ได้ติดตั้ง ให้รัน -> pip install fastapi uvicorn pydantic scikit-learn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
import numpy as np
from sklearn.linear_model import LogisticRegression

# 1. จำลองการฝึกสอนโมเดลไว้ในความจำ (In-memory Model)
X_mock = np.array([[20, 2], [25, 3], [35, 8], [45, 15], [50, 20]]) # [อายุ, ประสบการณ์]
y_mock = np.array([0, 0, 1, 1, 1]) # 0 = ไม่ผ่านเกณฑ์, 1 = ผ่านเกณฑ์โปรโมท
model = LogisticRegression()
model.fit(X_mock, y_mock)

# 2. กำหนดโครงสร้างข้อมูลขาเข้าด้วย Pydantic (Data Validation)
class EmployeeData(BaseModel):
    age: int = Field(..., ge=18, le=65, description="อายุของพนักงาน (18-65 ปี)")
    experience: float = Field(..., ge=0.0, description="ประสบการณ์ทำงาน (ปี)")

    class Config:
        schema_extra = {
            "example": {"age": 30, "experience": 5.5}
        }

# 3. สร้างแอป FastAPI
app = FastAPI(
    title="ML Model Serving API",
    description="API สำหรับทำนายโอกาสโปรโมทพนักงาน (Promotion Prediction Service)",
    version="1.0.0"
)

@app.get("/")
def health_check():
    """ตรวจสอบสถานะของ API"""
    return {"status": "healthy", "service": "ML Promotion Predictor V1"}

@app.post("/predict")
def predict_promotion(data: EmployeeData):
    """รับข้อมูลพนักงานมาคำนวณผลทำนายจากโมเดล"""
    try:
        # เตรียมฟีเจอร์เข้าโมเดล
        features = np.array([[data.age, data.experience]])
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1] # ความน่าจะเป็นที่จะเป็นคลาส 1

        result_text = "🎉 มีโอกาสสูงที่จะได้โปรโมท (Promoted)" if prediction == 1 else "⏳ ยังไม่ผ่านเกณฑ์ในรอบนี้"

        return {
            "input": {"age": data.age, "experience": data.experience},
            "prediction_code": int(prediction),
            "result_summary": result_text,
            "confidence_score": f"{probability * 100:.2f}%"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {str(e)}")

# --- วิธีการรัน API Server ใน Terminal ---
# คำสั่งรัน: uvicorn filename:app --reload --port 8000
# เข้าหน้าเอกสาร API อัตโนมัติ (Swagger UI): http://localhost:8000/docs
```

---

## 🔗 อ้างอิงและจุดเชื่อมโยง (Wiki-Links & Next Steps)
- **บทเรียนก่อนหน้า:** [[Week 3 - Trees, Ensembles & Unsupervised Learning]]
- **ภาพรวมเฟส:** [[Phase 1 - Overview|Phase 1: Foundations & Classical Machine Learning]]
- **ก้าวสู่เฟสที่ 2:** [[Phase 2 - Overview|Phase 2: Deep Learning & Unstructured Data]]
- **ความรู้ที่เกี่ยวข้อง:** [[FastAPI Documentation]], [[Python Multiprocessing Guide]], [[Design Patterns for ML]]
