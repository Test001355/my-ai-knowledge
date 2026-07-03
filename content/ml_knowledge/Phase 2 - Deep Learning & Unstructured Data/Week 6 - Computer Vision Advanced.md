---
title: Week 6 - Computer Vision Advanced
phase: Phase 2 - Deep Learning & Unstructured Data
tags: #ComputerVision #CNN #ResNet #TransferLearning #DataAugmentation #GradCAM #PyTorch #Segmentation
date: 2026-07-03
---

# 🟡 Week 6 - Computer Vision Advanced

> [!NOTE] 📋 ภาพรวมเนื้อหา (Overview)
> **ทฤษฎี (2 ชม.):** สถาปัตยกรรม CNN, ResNet, Transfer Learning, Data Augmentation  
> **เวิร์กชอป (Workshops):**
> - **Workshop 1:** การใช้ Pre-trained CNN Model ทำ Transfer Learning
> - **Workshop 2:** การทำ Data Augmentation เพื่อแก้ปัญหาข้อมูลภาพน้อย
> - **Workshop 3:** สร้างโมเดล Image Segmentation (แยกวัตถุตามพิกเซล)
> - **Workshop 4:** เทคนิคการตีความหมายโมเดลภาพ (Visualizing CNNs - Grad-CAM)

---

## 📖 ส่วนที่ 1: สรุปทฤษฎีสำคัญ (Key Theory Concepts)

### 1. Convolutional Neural Networks (CNNs)
โมเดลที่ถูกออกแบบมาเพื่อประมวลผลข้อมูลที่มีโครงสร้างเป็นตาราง 2D หรือ 3D เช่น รูปภาพ โดยใช้ตัวกรอง (Filters / Kernels) สแกนทั่วภาพเพื่อสกัดคุณลักษณะ (Feature Extraction) เช่น เส้นขอบ มุม หรือพื้นผิว
- **Convolutional Layer:** ใช้ Kernel คูณผลรวมกับพิกเซลภาพ เพื่อสกัดฟีเจอร์
- **Pooling Layer (Max / Average):** ลดขนาดมิติของภาพ (Downsampling) เพื่อลดปริมาณการคำนวณและป้องกัน Overfitting

### 2. ResNet & Transfer Learning
- **ResNet (Residual Network):** สถาปัตยกรรมที่แก้ปัญหา Vanishing Gradient ในเครือข่ายที่ลึกมาก ๆ ด้วยการใช้ทางลัด (Skip Connection / Residual Connection) ให้ข้อมูลกระโดดข้ามชั้นได้
- **Transfer Learning:** การนำโมเดลที่ฝึกสอนมาแล้วกับข้อมูลภาพมหาศาล (เช่น ImageNet 14 ล้านภาพ) มาปรับใช้กับงานของเรา โดยเปลี่ยนแค่ชั้นตัดสินใจสุดท้าย (Classification Head) ช่วยให้ใช้ภาพฝึกน้อยลงมากและแม่นยำสูง

---

## 🛠️ ส่วนที่ 2: เวิร์กชอปเชิงปฏิบัติการ (Hands-on Workshops)

### Workshop 1: การใช้ Pre-trained CNN Model ทำ Transfer Learning
โหลดโมเดล ResNet18 ที่ผ่านการฝึกสอนแล้วจาก `torchvision.models` มาปรับแต่งเพื่อจำแนกภาพ 2 คลาสใหม่ (เช่น แมว vs สุนัข)

```python
import torch
import torch.nn as nn
from torchvision import models

# 1. โหลดโมเดล Pre-trained ResNet18 (ใช้ Weights จาก ImageNet)
weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)

# 2. แช่แข็งพารามิเตอร์ (Freeze Weights) เพื่อไม่ให้ความรู้เดิมหายไปตอน Train
for param in model.parameters():
    param.requires_grad = False

# 3. ดูโครงสร้างชั้นสุดท้าย (fc layer เดิมมี 1,000 คลาสสำหรับ ImageNet)
num_ftrs = model.fc.in_features
print("จำนวนฟีเจอร์ก่อนเข้าชั้นตัดสินใจสุดท้าย:", num_ftrs)

# 4. เปลี่ยนชั้นสุดท้ายให้เหลือเพียง 2 คลาสตามโปรเจกต์ของเรา (Cat vs Dog)
model.fc = nn.Linear(num_ftrs, 2)

# ตอนนี้พารามิเตอร์ของ model.fc จะเป็น requires_grad=True โดยอัตโนมัติ
print("พร้อมสำหรับการ Fine-tune ชั้น fc แล้ว!")
```

---

### Workshop 2: การทำ Data Augmentation เพื่อแก้ปัญหาข้อมูลภาพน้อย
ใช้ `torchvision.transforms` สร้างความหลากหลายให้รูปภาพ (เช่น หมุน พลิก ปรับสี) เพื่อเพิ่มความทนทานให้โมเดลและป้องกัน Overfitting

```python
from torchvision import transforms
from PIL import Image
import numpy as np

# 1. กำหนด Pipeline สำหรับ Data Augmentation (ใช้ตอน Training)
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)), # ตัดขยายแบบสุ่ม
    transforms.RandomHorizontalFlip(p=0.5),                   # พลิกซ้าย-ขวา โอกาส 50%
    transforms.RandomRotation(degrees=15),                    # หมุนภาพสุ่มไม่เกิน 15 องศา
    transforms.ColorJitter(brightness=0.2, contrast=0.2),     # ปรับแสงและคอนทราสต์
    transforms.ToTensor(),                                    # แปลงเป็น PyTorch Tensor (0-1)
    transforms.Normalize(mean=[0.485, 0.456, 0.406],          # ปรับค่ามาตรฐาน ImageNet
                         std=[0.229, 0.224, 0.225])
])

# 2. กำหนด Pipeline สำหรับ Validation/Testing (ไม่ต้องสุ่ม หมุน หรือพลิก)
val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

print("Data Augmentation Pipeline พร้อมใช้งานกับ DataLoader!")
```

---

### Workshop 3: สร้างโมเดล Image Segmentation (แยกวัตถุตามพิกเซล)
ทดลองใช้โมเดล Semantic Segmentation อย่าง **DeepLabV3** เพื่อแยกพิกเซลคนหรือวัตถุออกจากพื้นหลัง

```python
import torch
from torchvision import models, transforms
import numpy as np

# 1. โหลดโมเดล DeepLabV3 Pre-trained บน COCO Dataset (แยกได้ 21 คลาส)
weights = models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT
seg_model = models.segmentation.deeplabv3_resnet50(weights=weights).eval()

# 2. จำลองสร้างภาพ Tensor สุ่มขนาด (3, 224, 224)
dummy_img = torch.randn(1, 3, 224, 224)

# 3. คำนวณพยากรณ์พิกเซล (Inference)
with torch.no_grad():
    output = seg_model(dummy_img)['out'] # output shape: (1, 21, 224, 224)

# 4. หาคลาสที่ได้คะแนนสูงสุดในแต่ละพิกเซล (Argmax)
seg_map = torch.argmax(output.squeeze(), dim=0).numpy()

print("ขนาดของ Segmentation Map:", seg_map.shape) # ควรได้ (224, 224)
print("คลาสที่พบในภาพจำลอง (0 = Background):", np.unique(seg_map))
```

---

### Workshop 4: เทคนิคการตีความหมายโมเดลภาพ (Visualizing CNNs - Grad-CAM)
ใช้เทคนิค **Grad-CAM (Gradient-weighted Class Activation Mapping)** สร้าง Heatmap สีแดง-น้ำเงิน เพื่อดูว่าโมเดล CNN "มองเห็นและให้ความสนใจ" พิกเซลส่วนไหนตอนตัดสินใจ

```python
# หมายเหตุ: หากยังไม่ได้ติดตั้ง ให้รัน -> pip install grad-cam
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
    has_cam = True
except ImportError:
    has_cam = False
    print("กรุณาติดตั้ง pytorch-grad-cam โดยรัน: pip install grad-cam")

if has_cam:
    # 1. โหลดโมเดล ResNet50
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).eval()
    
    # 2. เลือกชั้น Layer สุดท้ายของ Convolution (layer4[-1])
    target_layers = [model.layer4[-1]]
    
    # 3. สร้างอ็อบเจกต์ GradCAM
    cam = GradCAM(model=model, target_layers=target_layers)
    
    # 4. จำลอง Tensor ข้อมูลเข้า
    input_tensor = torch.randn(1, 3, 224, 224)
    
    # 5. คำนวณ Heatmap สำหรับคลาสเป้าหมาย
    targets = [ClassifierOutputTarget(281)] # 281 คือรหัสคลาสแมว (Tabby cat) ใน ImageNet
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
    
    print("คำนวณ Grad-CAM Heatmap สำเร็จ! ขนาด:", grayscale_cam.shape)
    print("สามารถนำ grayscale_cam ไปซ้อนทับภาพเดิมด้วย show_cam_on_image() เพื่อดูจุดสีแดงที่โมเดลสนใจได้ทันที")
```

---

## 🔗 อ้างอิงและจุดเชื่อมโยง (Wiki-Links & Next Steps)
- **บทเรียนก่อนหน้า:** [[Week 5 - Introduction to Deep Learning (PyTorch & FastAI)]]
- **บทเรียนถัดไป:** [[Week 7 - Natural Language Processing (NLP) Foundations]]
- **ภาพรวมเฟส:** [[Phase 2 - Overview|Phase 2: Deep Learning & Unstructured Data]]
