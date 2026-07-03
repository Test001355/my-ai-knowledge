---
title: Phase 4 - Production, MLOps & System Design
tags: #Phase4 #MLOps #SystemDesign #ModelServing #Docker #FastAPI #Monitoring #EvidentlyAI #MLflow
created: 2026-07-03
---

# 🟢 Phase 4: Production, MLOps & System Design

> [!INFO] 📌 เป้าหมายของ Phase 4 (สัปดาห์ที่ 12-14)
> ในเฟสที่สี่นี้จะเปลี่ยนคุณจาก Data Scientist ให้กลายเป็น **Machine Learning Engineer อย่างแท้จริง** โดยเน้นที่การนำโมเดลออกจาก Jupyter Notebook ไปติดตั้งใช้งานจริงบนระบบคลาวด์ (Model Serving & Deployment), การออกแบบสถาปัตยกรรมระดับ Enterprise (ML System Design) และการวางระบบติดตามดูแลรักษาโมเดลอัตโนมัติ **(MLOps & Continuous Training)**

---

## 🗓️ ตารางเรียนและเนื้อหาใน Phase 4

### [[Week 12 - Designing Machine Learning Systems]]
* **ทฤษฎี (2 ชม.):** การออกแบบสถาปัตยกรรม ML, Batch vs Real-time Serving, Feature Stores
* **Workshop 1:** ออกแบบ Data Pipeline สำหรับ ML System (ETL vs ELT)
* **Workshop 2:** จำลองระบบ Feature Store เบื้องต้นด้วย Redis / In-memory Dict
* **Workshop 3:** เปรียบเทียบสถาปัตยกรรม Batch Inference กับ Online Real-time Inference
* **Workshop 4:** การจัดการ Model Registry ด้วย MLflow เบื้องต้น

---

### [[Week 13 - Model Deployment & Serving]]
* **ทฤษฎี (2 ชม.):** การทำ Containerization ด้วย Docker, Serving Frameworks (Triton / TorchServe), ONNX
* **Workshop 1:** สร้าง Dockerfile สำหรับห่อหุ้ม ML Service ให้รันได้ทุกที่
* **Workshop 2:** การแปลงโมเดล PyTorch เป็นรูปแบบ ONNX เพื่อเพิ่มความเร็วในการ Inference
* **Workshop 3:** การเขียน Load Testing ด้วย Locust เพื่อทดสอบรับโหลดผู้ใช้พร้อมกัน
* **Workshop 4:** จำลองการ Deploy โมเดลบน Cloud (AWS ECS / Google Cloud Run)

---

### [[Week 14 - MLOps, Monitoring & Maintenance]]
* **ทฤษฎี (2 ชม.):** การเสื่อมสภาพของโมเดล (Model Decay / Drift), CI/CD/CT ใน MLOps
* **Workshop 1:** การใช้ Evidently AI ตรวจจับ Data Drift และ Concept Drift
* **Workshop 2:** สร้าง Automated Pipeline สำหรับ Retraining โมเดลเมื่อประสิทธิภาพตก
* **Workshop 3:** การส่งออก Metrics เพื่อแสดงผลบน Dashboard (Prometheus & Grafana)
* **Workshop 4:** สรุปแนวปฏิบัติการทำ Continuous Training (CT) ในระบบ MLOps

---

## 🔗 การเชื่อมโยงไปยัง Phase ถัดไป
- **กลับไปหน้าหลัก:** [[Home]]
- **ย้อนกลับไปเฟสที่ 3:** [[Phase 3 - Overview|Phase 3: Generative AI & Application Building]]
- **ไปต่อเฟสสุดท้าย:** [[Phase 5 - Overview|Phase 5: Capstone Project]]
