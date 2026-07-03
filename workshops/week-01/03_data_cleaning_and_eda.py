# ====================================================================
# Workshop 3: Data Cleaning & EDA
# ====================================================================

# # Workshop 3: การทำ Data Cleaning และ EDA (Exploratory Data Analysis) เบื้องต้น
# เรียนรู้เทคนิคการทำความสะอาดข้อมูลที่สูญหาย (Missing Values) และการสำรวจข้อมูลด้วยกราฟ

# ## 1. โหลดข้อมูลและตรวจสอบค่าว่าง (Missing Values Detection)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# สร้างข้อมูลจำลองที่มีปัญหา (Dirty Data)
np.random.seed(42)
n_samples = 100
data = {
    'Age': np.random.normal(35, 10, n_samples),
    'Income': np.random.normal(60000, 15000, n_samples),
    'Score': np.random.randint(1, 100, n_samples).astype(float),
    'Category': np.random.choice(['A', 'B', 'C', None], n_samples, p=[0.4, 0.3, 0.2, 0.1])
}

df = pd.DataFrame(data)
# ใส่ค่าว่างสุ่มใน Age และ Score
df.loc[df.sample(10).index, 'Age'] = np.nan
df.loc[df.sample(5).index, 'Score'] = np.nan

print("--- ตรวจสอบจำนวนค่าว่างในแต่ละคอลัมน์ ---")
print(df.isnull().sum())
print(f"\nจำนวนข้อมูลทั้งหมด: {len(df)} แถว")

# ------------------------------------------------------------

# ## 2. การทำความสะอาดข้อมูล (Data Cleaning & Imputation)

# 1. เติมค่าว่างใน 'Age' ด้วยค่ามัธยฐาน (Median) เพื่อป้องกันอิทธิพลจาก Outlier
age_median = df['Age'].median()
df['Age'].fillna(age_median, inplace=True)

# 2. เติมค่าว่างใน 'Score' ด้วยค่าเฉลี่ย (Mean)
score_mean = df['Score'].mean()
df['Score'].fillna(score_mean, inplace=True)

# 3. เติมค่าว่างใน 'Category' ด้วยฐานนิยม (Mode หรือ Unknown)
df['Category'].fillna('Unknown', inplace=True)

print("--- ตรวจสอบค่าว่างหลังทำความสะอาด ---")
print(df.isnull().sum())
print("\n✅ Data Cleaning เสร็จสมบูรณ์! ไม่มีค่าว่างเหลืออยู่ในตารางแล้ว")

# ------------------------------------------------------------

# ## 3. การสำรวจข้อมูลด้วยกราฟ (Exploratory Data Analysis - EDA)

# ตั้งค่าสไตล์กราฟ
sns.set_theme(style="whitegrid")

# สร้างหน้าต่างกราฟ 2x2
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. การกระจายตัวของอายุ (Distribution Plot)
sns.histplot(df['Age'], kde=True, color='skyblue', ax=axes[0, 0])
axes[0, 0].set_title('Age Distribution')

# 2. ความสัมพันธ์ระหว่าง Income กับ Score (Scatter Plot)
sns.scatterplot(data=df, x='Income', y='Score', hue='Category', s=80, ax=axes[0, 1])
axes[0, 1].set_title('Income vs Score by Category')

# 3. คะแนนเฉลี่ยแยกตามกลุ่ม Category (Bar Plot)
sns.barplot(data=df, x='Category', y='Score', palette='Set2', ax=axes[1, 0], ci=None)
axes[1, 0].set_title('Average Score by Category')

# 4. Boxplot ดูการกระจายและ Outlier ของ Income
sns.boxplot(y=df['Income'], color='lightgreen', ax=axes[1, 1])
axes[1, 1].set_title('Income Boxplot (Outlier Detection)')

plt.tight_layout()
# plt.show() # Uncomment เมื่อรันใน Jupyter Notebook
print("✅ EDA Visualizations Generated Successfully!")

# ------------------------------------------------------------
