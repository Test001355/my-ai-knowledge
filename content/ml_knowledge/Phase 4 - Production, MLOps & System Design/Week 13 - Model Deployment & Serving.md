---
title: Week 13 - Model Deployment & Serving
phase: Phase 4 - Production, MLOps & System Design
tags: #ModelServing #Docker #FastAPI #Triton #TorchScript #ONNX #LoadTesting #Locust
date: 2026-07-03
---

# 🟢 Week 13 - Model Deployment & Serving

> [!NOTE] 📋 ภาพรวมเนื้อหา (Overview)
> **ทฤษฎี (2 ชม.):** การทำ Containerization ด้วย Docker, Serving Frameworks (Triton / TorchServe), ONNX  
> **เวิร์กชอป (Workshops):**
> - **Workshop 1:** สร้าง Dockerfile สำหรับห่อหุ้ม ML Service ให้รันได้ทุกที่
> - **Workshop 2:** การแปลงโมเดล PyTorch เป็นรูปแบบ ONNX เพื่อเพิ่มความเร็วในการ Inference
> - **Workshop 3:** การเขียน Load Testing ด้วย Locust เพื่อทดสอบรับโหลดผู้ใช้พร้อมกัน
> - **Workshop 4:** จำลองการ Deploy โมเดลบน Cloud (AWS ECS / Google Cloud Run)

---

## 📖 ส่วนที่ 1: สรุปทฤษฎีสำคัญ (Key Theory Concepts)

### 1. ทำไมต้องใช้ Docker ในการ Deploy โมเดล?
ปัญหาคลาสสิก "มันรันได้บนเครื่องฉัน แต่พังบนเซิร์ฟเวอร์!" เกิดจากไลบรารีหรือเวอร์ชัน OS ไม่ตรงกัน การทำ **Containerization ด้วย Docker** จะห่อหุ้มทั้งโค้ด ไลบรารี Python และระบบปฏิบัติการเอาไว้ใน "ตู้คอนเทนเนอร์ (Image)" เดียว ทำให้มั่นใจได้ว่าจะรันได้เหมือนกัน 100% ทุกแพลตฟอร์ม

### 2. รูปแบบไฟล์ ONNX (Open Neural Network Exchange)
โมเดล PyTorch หรือ Scikit-Learn ทั่วไปมักมีขนาดใหญ่และกิน RAM สูง การแปลงเป็น **ONNX** ช่วยให้:
- ถอดการพึ่งพาไลบรารีเดิมออก (ไม่จำเป็นต้องลง PyTorch ก้อนโตบนเซิร์ฟเวอร์ปลายทาง)
- เร่งความเร็วในการพยากรณ์ (Inference Speed) ได้เร็วขึ้น 2 เท่า - 5 เท่า โดยใช้ **ONNX Runtime**

---

## 🛠️ ส่วนที่ 2: เวิร์กชอปเชิงปฏิบัติการ (Hands-on Workshops)

### Workshop 1: สร้าง Dockerfile สำหรับ ML Service
ตัวอย่างไฟล์ `Dockerfile` มาตรฐานและคำสั่งใน Terminal ในการสร้างและรัน Container สำหรับ ML Web API

```dockerfile
# ----------------------------------------------------
# ไฟล์ Dockerfile มาตรฐานสำหรับ FastAPI ML Service
# ----------------------------------------------------
# 1. ใช้ Base Image Python 3.10 ขนาดเล็ก (Slim)
FROM python:3.10-slim

# 2. ตั้งค่าแฟ้มทำงานภายใน Container
WORKDIR /app

# 3. คัดลอกไฟล์รายการไลบรารีไปติดตั้งก่อนเพื่อใช้ Cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. คัดลอกโค้ดโมเดลและแอปทั้งหมดเข้า Container
COPY . .

# 5. เปิดพอร์ต 8000 สำหรับสื่อสาร
EXPOSE 8000

# 6. คำสั่งเริ่มต้นรันเซิร์ฟเวอร์ Uvicorn เมื่อคอนเทนเนอร์เปิดทำงาน
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# --- วิธีการรัน Docker Command ใน Terminal ---
# สร้าง Docker Image ชื่อ ml-serving-api
docker build -t ml-serving-api:v1 .

# รัน Container ในพื้นหลัง (Detached mode) โดยจับคู่พอร์ต 8000 ของเครื่องเข้ากับ 8000 ของคอนเทนเนอร์
docker run -d -p 8000:8000 --name my_ml_api ml-serving-api:v1

# ตรวจสอบสถานะและดู Log
docker ps
docker logs -f my_ml_api
```

---

### Workshop 2: การแปลงโมเดล PyTorch เป็นรูปแบบ ONNX
เขียน Python สคริปต์เพื่อส่งออกโมเดล PyTorch เป็นไฟล์ `.onnx` และทดสอบรันด้วย ONNX Runtime

```python
# หมายเหตุ: หากยังไม่ได้ติดตั้ง ให้รัน -> pip install onnx onnxruntime torch
import torch
import torch.nn as nn
import numpy as np

# 1. สร้างโมเดล PyTorch จำลอง
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 2)
        )
    def forward(self, x):
        return self.fc(x)

torch_model = SimpleMLP().eval()

# 2. สร้างข้อมูลตัวอย่าง (Dummy Input) ขนาด (Batch=1, Features=10)
dummy_input = torch.randn(1, 10)

# 3. ส่งออกโมเดลเป็นไฟล์ ONNX (Exporting)
onnx_file_path = "simple_mlp.onnx"
torch.onnx.export(
    torch_model,
    dummy_input,
    onnx_file_path,
    input_names=["input_features"],
    output_names=["output_logits"],
    dynamic_axes={"input_features": {0: "batch_size"}, "output_logits": {0: "batch_size"}} # รองรับขนาด Batch ไม่ตายตัว
)
print(f"✅ แปลงโมเดล PyTorch เป็นไฟล์ ONNX สำเร็จ -> {onnx_file_path}")

# ----------------------------------------------------
# 4. ทดสอบโหลดและรันพยากรณ์ด้วย ONNX Runtime
# ----------------------------------------------------
try:
    import onnxruntime as ort
    session = ort.InferenceSession(onnx_file_path)
    
    # เตรียมข้อมูลเข้าเป็น NumPy array ชนิด float32
    input_data = np.random.randn(5, 10).astype(np.float32) # ทดสอบ Batch=5
    
    # คำนวณผลทำนายด้วย ONNX Runtime
    onnx_outputs = session.run(None, {"input_features": input_data})[0]
    print("\n⚡ ผลการรัน Inference เร็วสูงด้วย ONNX Runtime (Batch Size=5):")
    print(np.round(onnx_outputs, 3))
except ImportError:
    print("ติดตั้ง onnxruntime เพื่อรันสอบโมเดล ONNX ได้อย่างรวดเร็ว")
```

---

### Workshop 3: การเขียน Load Testing ด้วย Locust
สร้างสคริปต์ **Locust** เพื่อยิงคำขอกระหน่ำเข้าใส่ ML API ของเรา เพื่อดูว่าจะรองรับผู้ใช้งานพร้อมกันได้สูงสุดกี่คนก่อนระบบจะล่ม

```python
# หมายเหตุ: หากยังไม่ได้ติดตั้ง ให้รัน -> pip install locust
# ไฟล์นี้ควรบันทึกชื่อ locustfile.py
from locust import HttpUser, task, between
import json
import random

class MLUserBehavior(HttpUser):
    # สุ่มรอเวลาระหว่างคำขอแต่ละครั้ง 0.5 ถึง 2 วินาที (จำลองพฤติกรรมมนุษย์จริง)
    wait_time = between(0.5, 2.0)

    @task(3)
    def predict_promotion(self):
        """จำลองการยิง Request ขอผลทำนาย (น้ำหนักความถี่ = 3 เท่า)"""
        payload = {
            "age": random.randint(22, 55),
            "experience": round(random.uniform(1.0, 20.0), 1)
        }
        headers = {"Content-Type": "application/json"}
        
        with self.client.post("/predict", data=json.dumps(payload), headers=headers, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed with status {response.status_code}")

    @task(1)
    def check_health(self):
        """จำลองการตรวจสอบ Health Check (น้ำหนักความถี่ = 1 เท่า)"""
        self.client.get("/")

# --- วิธีรัน Load Test ใน Terminal ---
# รันคำสั่ง: locust -f locustfile.py
# เปิดเบราว์เซอร์ไปที่ http://localhost:8089 เพื่อใส่จำนวน Users และเริ่มทดสอบทันที!
print("สคริปต์ Locust พร้อมแล้ว! รัน 'locust -f locustfile.py' ใน Terminal เพื่อทดสอบรับโหลด")
```

---

### Workshop 4: จำลองการ Deploy โมเดลบน Cloud (AWS / GCP)
ตารางสรุปบริการ Cloud ชั้นนำสำหรับการรัน Docker Container ของ ML Service

| บริการ Cloud | ประเภท | ความยากง่าย | ความเหมาะสมในการใช้งาน |
| :--- | :--- | :--- | :--- |
| **Google Cloud Run** | Serverless Container | ⭐ ง่ายมาก | เหมาะกับ API ทั่วไป คิดเงินเฉพาะวินาทีที่มีคนยิง Request ปรับสเกลเป็น 0 ได้เมื่อไม่มีคนใช้ |
| **AWS ECS (Fargate)** | Container Service | ⭐⭐ ปานกลาง | เหมาะกับระบบองค์กรขนาดใหญ่ที่มี Microservices หลายตัวคุยกันใน VPC |
| **AWS SageMaker Endpoints**| Dedicated ML Serving | ⭐⭐⭐ สูง | เหมาะกับโมเดล Deep Learning / LLM ที่ต้องการใช้ GPU Cluster พร้อมระบบ Monitoring ครบวงจร |
| **Hugging Face Spaces** | ML Demo Platform | ⭐ ง่ายที่สุด | เหมาะสำหรับการจัดทำ Demo โชว์ผลงานด้วย Gradio หรือ Streamlit ให้ลูกค้าดูอย่างรวดเร็ว |

---

## 🔗 อ้างอิงและจุดเชื่อมโยง (Wiki-Links & Next Steps)
- **บทเรียนก่อนหน้า:** [[Week 12 - Designing Machine Learning Systems]]
- **ภาพรวมเฟส:** [[Phase 4 - Overview|Phase 4: Production, MLOps & System Design]]
- **บทเรียนถัดไป:** [[Week 14 - MLOps, Monitoring & Maintenance]]
- **ความรู้ที่เกี่ยวข้อง:** [[Docker for Data Scientists Guide]], [[ONNX Runtime Performance]], [[Cloud Serving Best Practices]]
