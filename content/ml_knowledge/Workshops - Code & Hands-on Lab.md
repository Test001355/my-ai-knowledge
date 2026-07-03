---
title: Workshops - Code & Hands-on Lab (Week 1 & Week 2)
tags: #Workshops #Jupyter #Python #Code #HandsOn #MachineLearning
date: 2026-07-03
---

# 🛠️ Workshops - Code & Hands-on Lab (Week 1 & Week 2)

> [!TIP] 📥 ดาวน์โหลดโค้ดและรันจริง (How to Download & Run)
> โค้ดเวิร์กชอปทั้งหมดในหลักสูตรถูกจัดเก็บไว้ในโฟลเดอร์ `workshops/` ของ GitHub Repository นี้ คุณสามารถ Clone โครงการไปรันบนเครื่องของคุณหรือเปิดใน Jupyter Notebook และ Google Colab ได้ทันที!
> 
> **ลิงก์ GitHub Repository:** [https://github.com/Test001355/my-ai-knowledge](https://github.com/Test001355/my-ai-knowledge)

---

## 📂 โครงสร้างเวิร์กชอปที่มีใน Repository

โค้ดทุกเวิร์กชอปถูกเตรียมไว้ให้ทั้ง 2 รูปแบบ:
1. **Jupyter Notebook (`.ipynb`)**: เหมาะสำหรับเปิดอ่านและทดลองรันทีละขั้นตอน (Cell by Cell) พร้อมกราฟและผลลัพธ์
2. **Python Script (`.py`)**: เหมาะสำหรับรันผ่าน Command Line / Terminal หรือนำไปต่อยอดในโปรเจกต์จริง

### 🟢 Week 1: Introduction to ML & Software Engineering Basics
| เวิร์กชอป | หัวข้อการเรียนรู้ | ไฟล์ใน Repository |
| :--- | :--- | :--- |
| **Workshop 1** | การตั้งค่า Environment (Conda/Venv) และ Git Version Control | `workshops/week-01/01_env_and_git.ipynb` / `.py` |
| **Workshop 2** | Data Manipulation พื้นฐานด้วย Pandas & NumPy | `workshops/week-01/02_data_manipulation_pandas_numpy.ipynb` / `.py` |
| **Workshop 3** | Data Cleaning และ Exploratory Data Analysis (EDA) เบื้องต้น | `workshops/week-01/03_data_cleaning_and_eda.ipynb` / `.py` |
| **Workshop 4** | สร้างโมเดลแรกด้วย Scikit-Learn (Linear Regression) | `workshops/week-01/04_first_linear_regression_model.ipynb` / `.py` |

---

### 🔵 Week 2: Classification, Metrics & Model Selection
| เวิร์กชอป | หัวข้อการเรียนรู้ | ไฟล์ใน Repository |
| :--- | :--- | :--- |
| **Workshop 1** | การสร้างโมเดล Binary Classification (Logistic Regression / SGD) | `workshops/week-02/01_binary_classification_logistic_sgd.ipynb` / `.py` |
| **Workshop 2** | การจัดการข้อมูลแบบ Multiclass และสร้าง Confusion Matrix | `workshops/week-02/02_multiclass_classification_confusion_matrix.ipynb` / `.py` |
| **Workshop 3** | การทำ Cross-Validation และ Hyperparameter Tuning (GridSearch) | `workshops/week-02/03_cross_validation_and_gridsearch.ipynb` / `.py` |
| **Workshop 4** | การใช้ Support Vector Machines (SVM) และเปรียบเทียบประสิทธิภาพ | `workshops/week-02/04_support_vector_machines_svm.ipynb` / `.py` |

---

## 🚀 ขั้นตอนการเริ่มต้นใช้งาน (Quick Start Guide)

### 1. โคลนคลังโค้ดลงเครื่อง (Clone Repository)
เปิด Terminal หรือ Command Prompt แล้วรันคำสั่ง:
```bash
git clone https://github.com/Test001355/my-ai-knowledge.git
cd my-ai-knowledge
```

### 2. ติดตั้งไลบรารีที่จำเป็น (Install Dependencies)
```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
```

### 3. เปิดใช้งาน Jupyter Notebook
```bash
jupyter notebook
```
เมื่อหน้าเว็บ Jupyter Lab / Notebook เปิดขึ้นมา ให้คลิกเข้าไปที่โฟลเดอร์ `workshops/` จากนั้นเลือกสัปดาห์ที่ต้องการ (`week-01` หรือ `week-02`) และคลิกเปิดไฟล์ `.ipynb` ที่สนใจได้ทันที!

---

## 🔗 อ้างอิงและจุดเชื่อมโยง (Wiki-Links)
- **หน้าหลักสูตร:** [[Home|สารบัญหลักสูตร AI & ML Engineering]]
- **เนื้อหาสัปดาห์ที่ 1:** [[Week 1 - Introduction to ML & Software Engineering Basics]]
- **เนื้อหาสัปดาห์ที่ 2:** [[Week 2 - Classification, Metrics & Model Selection]]
