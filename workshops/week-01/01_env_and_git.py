# ====================================================================
# Workshop 1: Environment & Git Setup
# ====================================================================

# # Workshop 1: การตั้งค่า Environment และ Git Version Control
# ในเวิร์กชอปนี้ เราจะเรียนรู้ขั้นตอนการตั้งค่า Python Environment และการใช้ Git ในการควบคุมเวอร์ชันของโปรเจกต์ Machine Learning

# ## 1. การสร้างและจัดการ Virtual Environment (Conda & Venv)
# การแยก Environment ช่วยป้องกันไม่ให้เวอร์ชันของไลบรารีในแต่ละโปรเจกต์ตีกัน

# คำสั่งสำหรับรันใน Terminal / Command Prompt (ไม่ต้องรันใน Python)

'''
# --- สำหรับ Conda ---
# 1. สร้าง environment ใหม่ชื่อ ml_env พร้อมติดตั้ง Python 3.10
conda create -n ml_env python=3.10 -y

# 2. เปิดใช้งาน environment
conda activate ml_env

# 3. ติดตั้งไลบรารีสำคัญสำหรับ Machine Learning
conda install numpy pandas scikit-learn matplotlib seaborn jupyter -y


# --- สำหรับ Python Venv (มาตรฐาน) ---
# 1. สร้าง environment ใหม่
python -m venv ml_env

# 2. เปิดใช้งานใน Windows
ml_env\Scripts\activate

# 3. เปิดใช้งานใน macOS / Linux
source ml_env/bin/activate

# 4. ติดตั้งไลบรารีผ่าน pip
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
'''
print("✅ Environment setup instructions reviewed successfully!")

# ------------------------------------------------------------

# ## 2. การใช้ Git Version Control ในโปรเจกต์ ML
# ขั้นตอนมาตรฐานในการเก็บโค้ดและอัปโหลดขึ้น GitHub

# คำสั่ง Git พื้นฐานสำหรับรันใน Terminal

'''
# 1. เริ่มต้นระบบ Git ในโฟลเดอร์โปรเจกต์
git init

# 2. สร้างไฟล์ .gitignore เพื่อไม่ให้ Git จำโฟลเดอร์ที่ไม่จำเป็น (เช่น ml_env, __pycache__)
echo "ml_env/" > .gitignore
echo ".ipynb_checkpoints/" >> .gitignore
echo "__pycache__/" >> .gitignore

# 3. ตรวจสอบสถานะไฟล์
git status

# 4. เพิ่มไฟล์ทั้งหมดเตรียมบันทึก
git add .

# 5. บันทึกเวอร์ชันโค้ด (Commit)
git commit -m "feat: initial setup for ML workshop"

# 6. เชื่อมต่อกับ GitHub Repository และ Push Код
git branch -M main
# git remote add origin https://github.com/username/my-ml-project.git
# git push -u origin main
'''
print("✅ Git version control workflow reviewed successfully!")

# ------------------------------------------------------------
