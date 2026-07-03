# ====================================================================
# Workshop 2: Pandas & NumPy Basics
# ====================================================================

# # Workshop 2: Data Manipulation พื้นฐานด้วย Pandas & NumPy
# เรียนรู้การจัดการตารางข้อมูลและตัวเลขความเร็วสูง ซึ่งเป็นหัวใจสำคัญของ Data Science

# ## 1. NumPy: Numerical Python
# การทำงานกับ Array และ Matrix สำหรับการคำนวณทางคณิตศาสตร์

import numpy as np

# สร้าง Array 1 มิติ และ 2 มิติ
arr_1d = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

print("--- NumPy Array Statistics ---")
print("Array 1D:", arr_1d)
print("Mean (ค่าเฉลี่ย):", np.mean(arr_1d))
print("Std (ค่าเบี่ยงเบนมาตรฐาน):", np.std(arr_1d))
print("Max / Min:", np.max(arr_1d), "/", np.min(arr_1d))
print("\nMatrix 2D Shape:", arr_2d.shape)
print("Matrix Dot Product:\n", np.dot(arr_2d, arr_2d))

# ------------------------------------------------------------

# ## 2. Pandas: Data Manipulation & Analysis
# การจัดการข้อมูลแบบตาราง (DataFrame) และการวิเคราะห์เบื้องต้น

import pandas as pd
import numpy as np

# สร้าง DataFrame จำลองข้อมูลพนักงานและเงินเดือน
data = {
    'Employee_ID': ['E001', 'E002', 'E003', 'E004', 'E005', 'E006'],
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva', 'Frank'],
    'Department': ['IT', 'HR', 'IT', 'Marketing', 'IT', 'HR'],
    'Age': [25, 30, 35, 28, 40, np.nan],  # มีค่าว่าง (NaN) 1 ตำแหน่ง
    'Salary': [50000, 60000, 75000, 65000, 90000, 55000],
    'Years_Experience': [2, 5, 8, 3, 12, 1]
}

df = pd.DataFrame(data)
print("--- ข้อมูลพนักงาน (DataFrame Head) ---")
print(df)

print("\n--- สถิติพื้นฐานของข้อมูลตัวเลข (Describe) ---")
print(df.describe())

print("\n--- เงินเดือนเฉลี่ยแยกตามแผนก (Groupby Department) ---")
print(df.groupby('Department')['Salary'].mean().reset_index())

# ------------------------------------------------------------
