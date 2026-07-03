---
title: Week 5 - Introduction to Deep Learning (PyTorch & FastAI)
phase: Phase 2 - Deep Learning & Unstructured Data
tags: #DeepLearning #PyTorch #FastAI #NeuralNetworks #GradientDescent #Autograd #Gradio
date: 2026-07-03
---

# 🟡 Week 5 - Introduction to Deep Learning (PyTorch & FastAI)

> [!NOTE] 📋 ภาพรวมเนื้อหา (Overview)
> **ทฤษฎี (2 ชม.):** พื้นฐาน Neural Networks, Gradient Descent, Backpropagation, แนะนำ PyTorch  
> **เวิร์กชอป (Workshops):**
> - **Workshop 1:** การจัดการ Tensors และ Autograd พื้นฐานใน PyTorch
> - **Workshop 2:** สร้าง Neural Network แบบง่าย (จากศูนย์) โดยไม่ใช้ Framework สำเร็จรูป
> - **Workshop 3:** สร้าง Image Classifier อย่างง่ายโดยใช้ FastAI Library
> - **Workshop 4:** นำโมเดลจาก Workshop 3 ไปสร้าง Web App บน Hugging Face Spaces (Gradio)

---

## 📖 ส่วนที่ 1: สรุปทฤษฎีสำคัญ (Key Theory Concepts)

### 1. พื้นฐานโครงข่ายประสาทเทียม (Artificial Neural Networks - ANN)
โครงข่ายประสาทเทียมจำลองการทำงานมาจากเซลล์ประสาทในสมองของมนุษย์ (Neuron) โดยประกอบด้วยชั้น (Layers) ต่าง ๆ ได้แก่:
- **Input Layer:** รับข้อมูลตัวเลข (Tensors) เข้ามาในระบบ
- **Hidden Layers:** ทำการคูณด้วยน้ำหนัก (Weights - $W$) บวกด้วยค่าอคติ (Bias - $b$) และผ่านฟังก์ชันกระตุ้น (Activation Function)
- **Output Layer:** ส่งออกผลลัพธ์การพยากรณ์ เช่น ค่าความน่าจะเป็น (Probability)

$$y = \sigma(W \cdot x + b)$$

### 2. Gradient Descent & Backpropagation
- **Loss Function:** ตัวชี้วัดว่าโมเดลทำนายผิดพลาดไปเท่าใด (เช่น Cross-Entropy Loss หรือ MSE)
- **Backpropagation:** การคำนวณหาค่าอนุพันธ์ย้อนกลับ (Chain Rule) เพื่อหาว่า Weight แต่ละตัวส่งผลต่อความผิดพลาดมากแค่ไหน
- **Gradient Descent:** การปรับน้ำหนัก (Weights Update) ในทิศทางตรงข้ามกับความชัน เพื่อให้ค่า Loss ลดลงต่ำที่สุด

---

## 🛠️ ส่วนที่ 2: เวิร์กชอปเชิงปฏิบัติการ (Hands-on Workshops)

### Workshop 1: การจัดการ Tensors และ Autograd พื้นฐานใน PyTorch
เรียนรู้โครงสร้างข้อมูล Tensor และระบบคำนวณอนุพันธ์อัตโนมัติ (Autograd) ซึ่งเป็นหัวใจสำคัญของ PyTorch

```python
import torch

# 1. สร้าง Tensors พื้นฐาน
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
print("Tensor x:\n", x)

# 2. ทำการคำนวณทางคณิตศาสตร์ (Forward Pass)
# y = x^2 + 3*x + 5
y = x**2 + 3*x + 5
out = y.mean()

print("\nผลลัพธ์ Forward Pass (out):", out.item())

# 3. คำนวณอนุพันธ์อัตโนมัติ (Backward Pass / Autograd)
# หา dy/dx จากสมการ 2x + 3
out.backward()

print("\nค่าอนุพันธ์ (Gradients - dx):")
print(x.grad) # ควรได้ค่าเท่ากับ 2*x + 3
```

---

### Workshop 2: สร้าง Neural Network แบบง่าย (จากศูนย์) โดยไม่ใช้ Framework สำเร็จรูป
ทำความเข้าใจคณิตศาสตร์เบื้องหลัง Deep Learning ด้วยการเขียน Perceptron และ Gradient Descent จากศูนย์ด้วย NumPy

```python
import numpy as np

# 1. เตรียมข้อมูลจำลอง (ตรรกศาสตร์ AND Gate)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [0], [0], [1]]) # เป็นจริง (1) เมื่อเป็น 1 ทั้งคู่เท่านั้น

# 2. กำหนด Hyperparameters และเริ่มต้นสุ่มน้ำหนัก (Weights & Bias)
np.random.seed(42)
weights = np.random.randn(2, 1) * 0.1
bias = np.random.randn(1) * 0.1
learning_rate = 0.5

# ฟังก์ชันกระตุ้น Sigmoid และอนุพันธ์ของมัน
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_deriv(a):
    return a * (1 - a)

# 3. เริ่มวนลูปฝึกสอน (Training Loop)
for epoch in range(1, 1001):
    # Forward Pass
    z = np.dot(X, weights) + bias
    predictions = sigmoid(z)
    
    # คำนวณข้อผิดพลาด (Loss / Error)
    error = predictions - y
    loss = np.mean(error**2)
    
    # Backward Pass (หา Gradients)
    d_loss = 2 * error / len(X)
    d_z = d_loss * sigmoid_deriv(predictions)
    d_weights = np.dot(X.T, d_z)
    d_bias = np.sum(d_z)
    
    # Update Weights & Bias (Gradient Descent Step)
    weights -= learning_rate * d_weights
    bias -= learning_rate * d_bias

    if epoch % 200 == 0:
        print(f"Epoch {epoch:04d} | Loss: {loss:.6f}")

print("\n--- ผลการทำนายของโครงข่ายประสาทจากศูนย์ ---")
print("Predictions:\n", np.round(sigmoid(np.dot(X, weights) + bias), 3))
```

---

### Workshop 3: สร้าง Image Classifier อย่างง่ายโดยใช้ FastAI Library
ใช้ FastAI ซึ่งเป็น High-Level API ของ PyTorch เพื่อฝึกโมเดลแยกประเภทภาพในไม่กี่บรรทัดด้วย Transfer Learning

```python
# หมายเหตุ: หากยังไม่ได้ติดตั้ง ให้รัน -> pip install fastai
try:
    from fastai.vision.all import *
    has_fastai = True
except ImportError:
    has_fastai = False
    print("กรุณาติดตั้ง fastai โดยรัน: pip install fastai")

if has_fastai:
    # 1. ดาวน์โหลดชุดข้อมูลภาพสัตว์เลี้ยง (Oxford-IIIT Pet Dataset)
    path = untar_data(URLs.PETS)/'images'
    
    # 2. กำหนดกติกาการดึงข้อมูล (DataBlock)
    # ฟังก์ชันตรวจสอบว่าภาพเป็นแมวหรือไม่ (ชื่อไฟล์แมวจะขึ้นต้นด้วยอักษรพิมพ์ใหญ่)
    def is_cat(x): return x[0].isupper()

    dls = ImageDataLoaders.from_name_func(
        path, get_image_files(path), valid_pct=0.2, seed=42,
        label_func=is_cat, item_tfms=Resize(224)
    )

    # 3. สร้างโมเดลโดยใช้ ResNet34 ทำ Transfer Learning
    learn = vision_learner(dls, resnet34, metrics=error_rate)
    
    # 4. เริ่มฝึกสอน (Fine-tune) 1 Epoch
    print("กำลัง Fine-tune โมเดลจำแนกภาพน้องแมว/น้องหมา...")
    learn.fine_tune(1)

    # 5. ทดสอบพยากรณ์ภาพ
    print("ฝึกสอนเสร็จสิ้น! สามารถใช้ learn.predict('image.jpg') เพื่อทดสอบทายภาพได้ทันที")
```

---

### Workshop 4: นำโมเดลจาก Workshop 3 ไปสร้าง Web App บน Hugging Face Spaces (Gradio)
สร้าง Web UI สวย ๆ เพื่อให้ผู้ใช้งานสามารถอัปโหลดรูปภาพเข้ามาให้ AI ทายผลได้ทันทีโดยใช้ไลบรารี Gradio

```python
# หมายเหตุ: หากยังไม่ได้ติดตั้ง ให้รัน -> pip install gradio
import gradio as gr
import numpy as np

def mock_image_classifier(image):
    """จำลองฟังก์ชันรับภาพแล้วคืนค่าความน่าจะเป็น (Inference Function)"""
    if image is None:
        return {"กรุณาอัปโหลดรูปภาพ": 1.0}
    
    # จำลองสุ่มผลลัพธ์ความมั่นใจของโมเดล
    prob_cat = np.random.uniform(0.1, 0.9)
    prob_dog = 1.0 - prob_cat
    
    return {
        "🐱 น้องแมว (Cat)": float(prob_cat),
        "🐶 น้องหมา (Dog)": float(prob_dog)
    }

# สร้างอินเทอร์เฟซด้วย Gradio
if __name__ == "__main__":
    interface = gr.Interface(
        fn=mock_image_classifier,
        inputs=gr.Image(type="numpy", label="อัปโหลดรูปภาพที่นี่"),
        outputs=gr.Label(num_top_classes=2, label="ผลการพยากรณ์จากโมเดล"),
        title="🤖 AI Pet Classifier Web App",
        description="แอปพลิเคชันจำแนกภาพแมวและสุนัข สร้างด้วย Gradio และ พร้อมอัปโหลดขึ้น Hugging Face Spaces"
    )
    
    # รันเซิร์ฟเวอร์ (ถ้าอยู่บน Hugging Face Spaces ระบบจะทำงานให้อัตโนมัติ)
    # interface.launch(share=True) # เปิด Share=True เพื่อได้ Public Link
    print("โค้ด Gradio UI พร้อมทำงาน! ใช้ interface.launch() เพื่อเปิดหน้าเว็บได้ทันที")
```

---

## 🔗 อ้างอิงและจุดเชื่อมโยง (Wiki-Links & Next Steps)
- **ภาพรวมเฟส:** [[Phase 2 - Overview|Phase 2: Deep Learning & Unstructured Data]]
- **บทเรียนถัดไป:** [[Week 6 - Computer Vision Advanced]]
- **ความรู้ที่เกี่ยวข้อง:** [[PyTorch Autograd Guide]], [[FastAI Transfer Learning]], [[Gradio Web Serving]]
