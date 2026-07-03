# 🛠️ Machine Learning Hands-on Workshops (Week 1 & Week 2)

ยินดีต้อนรับสู่คลังโค้ดเวิร์กชอปเชิงปฏิบัติการ (Hands-on Workshops) สำหรับหลักสูตร **Machine Learning 15 Weeks End-to-End** โค้ดทั้งหมดในโฟลเดอร์นี้ถูกเตรียมไว้ 2 รูปแบบ เพื่อความสะดวกสูงสุดในการเรียนรู้และการทดลองรันจริง:
1. **Jupyter Notebook (`.ipynb`)**: เหมาะสำหรับเปิดรันใน Jupyter Lab, VS Code หรือ Google Colab มีคำอธิบายและผลลัพธ์แยกเป็นเซลล์อย่างสวยงาม
2. **Python Script (`.py`)**: เหมาะสำหรับรันผ่าน Command Line / Terminal หรือนำไปต่อยอดในโปรเจกต์ซอฟต์แวร์จริง

---

## 📂 โครงสร้างเวิร์กชอป (Workshop Structure)

### 🟢 Week 1: Introduction to ML & Software Engineering Basics
| Workshop | รายละเอียดเนื้อหา | โฟลเดอร์ / ไฟล์ |
| :--- | :--- | :--- |
| **Workshop 1** | การตั้งค่า Environment (Conda/Venv) & Git Version Control | [01_env_and_git.ipynb](./week-01/01_env_and_git.ipynb) / [`.py`](./week-01/01_env_and_git.py) |
| **Workshop 2** | Data Manipulation พื้นฐานด้วย Pandas & NumPy | [02_data_manipulation_pandas_numpy.ipynb](./week-01/02_data_manipulation_pandas_numpy.ipynb) / [`.py`](./week-01/02_data_manipulation_pandas_numpy.py) |
| **Workshop 3** | Data Cleaning & Exploratory Data Analysis (EDA) เบื้องต้น | [03_data_cleaning_and_eda.ipynb](./week-01/03_data_cleaning_and_eda.ipynb) / [`.py`](./week-01/03_data_cleaning_and_eda.py) |
| **Workshop 4** | สร้างโมเดลแรกด้วย Scikit-Learn (Linear Regression) | [04_first_linear_regression_model.ipynb](./week-01/04_first_linear_regression_model.ipynb) / [`.py`](./week-01/04_first_linear_regression_model.py) |

### 🔵 Week 2: Classification, Metrics & Model Selection
| Workshop | รายละเอียดเนื้อหา | โฟลเดอร์ / ไฟล์ |
| :--- | :--- | :--- |
| **Workshop 1** | การสร้างโมเดล Binary Classification (Logistic Regression / SGD) | [01_binary_classification_logistic_sgd.ipynb](./week-02/01_binary_classification_logistic_sgd.ipynb) / [`.py`](./week-02/01_binary_classification_logistic_sgd.py) |
| **Workshop 2** | การจัดการข้อมูลแบบ Multiclass & สร้าง Confusion Matrix | [02_multiclass_classification_confusion_matrix.ipynb](./week-02/02_multiclass_classification_confusion_matrix.ipynb) / [`.py`](./week-02/02_multiclass_classification_confusion_matrix.py) |
| **Workshop 3** | การทำ Cross-Validation & Hyperparameter Tuning (GridSearch) | [03_cross_validation_and_gridsearch.ipynb](./week-02/03_cross_validation_and_gridsearch.ipynb) / [`.py`](./week-02/03_cross_validation_and_gridsearch.py) |
| **Workshop 4** | การใช้ Support Vector Machines (SVM) & เปรียบเทียบประสิทธิภาพ | [04_support_vector_machines_svm.ipynb](./week-02/04_support_vector_machines_svm.ipynb) / [`.py`](./week-02/04_support_vector_machines_svm.py) |

---

## 🚀 วิธีการใช้งานและการรันโค้ด (How to Run)

### 1. ติดตั้งไลบรารีที่จำเป็น
เปิด Terminal / Command Prompt ในโฟลเดอร์โปรเจกต์ แล้วรันคำสั่ง:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
```

### 2. วิธีรันผ่าน Jupyter Notebook (แนะนำ 🌟)
```bash
jupyter notebook
```
จากนั้นคลิกเข้าไปที่โฟลเดอร์ `workshops/week-01/` หรือ `workshops/week-02/` และเลือกไฟล์ `.ipynb` ที่ต้องการลองรันได้ทันที

### 3. วิธีรันผ่าน Python Script ใน Terminal
```bash
python workshops/week-01/02_data_manipulation_pandas_numpy.py
```
