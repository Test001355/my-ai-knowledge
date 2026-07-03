# ====================================================================
# Workshop 4: Scikit-Learn Linear Regression
# ====================================================================

# # Workshop 4: สร้างโมเดลแรกด้วย Scikit-Learn (Linear Regression)
# สร้างโมเดล Machine Learning แรกของคุณ เพื่อทำนายราคาบ้านจากขนาดพื้นที่ใช้สอย พร้อมประเมินผลความแม่นยำ

# ## 1. เตรียมข้อมูล Train/Test Split

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# สร้างข้อมูลจำลองความสัมพันธ์ระหว่าง พื้นที่บ้าน (ตร.ม.) และ ราคาบ้าน (ล้านบาท)
np.random.seed(42)
X_area = np.random.uniform(30, 250, 100).reshape(-1, 1)  # พื้นที่ 30 ถึง 250 ตร.ม.
# ราคา = 0.5 + (0.035 * พื้นที่) + สัญญาณรบกวน (Noise)
y_price = 0.5 + (0.035 * X_area.flatten()) + np.random.normal(0, 0.4, 100)

# แบ่งข้อมูลสำหรับ Train 80% และ Test 20%
X_train, X_test, y_train, y_test = train_test_split(X_area, y_price, test_size=0.2, random_state=42)

print(f"จำนวนข้อมูล Train: {len(X_train)} ตัวอย่าง")
print(f"จำนวนข้อมูล Test:  {len(X_test)} ตัวอย่าง")

# ------------------------------------------------------------

# ## 2. ฝึกสอนโมเดล Linear Regression (Model Training)

# สร้างและฝึกสอนโมเดล
model = LinearRegression()
model.fit(X_train, y_train)

# แสดงค่าสมการเส้นตรงที่โมเดลเรียนรู้ได้ (y = mx + c)
slope = model.coef_[0]
intercept = model.intercept_

print("--- ผลการเรียนรู้ของโมเดล ---")
print(f"สัมประสิทธิ์ความชัน (Slope / m): {slope:.4f} (ราคาเพิ่มขึ้น ~{slope*1000:.1f} พันบาท ต่อ ตร.ม.)")
print(f"จุดตัดแกน Y (Intercept / c):    {intercept:.4f} ล้านบาท")

# ------------------------------------------------------------

# ## 3. ประเมินประสิทธิภาพโมเดล (Model Evaluation)

# นำโมเดลไปทำนายข้อสอบ (Test Set)
y_pred = model.predict(X_test)

# คำนวณค่าสถิติวัดผล
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("--- ประสิทธิภาพของโมเดลบน Test Set ---")
print(f"Mean Absolute Error (MAE): {mae:.4f} ล้านบาท")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f} ล้านบาท")
print(f"R-squared Score (R²): {r2:.4f} (โมเดลอธิบายความผันแปรของราคาได้ {r2*100:.2f}%)")

# ------------------------------------------------------------

# ## 4. พล็อตเส้นแนวโน้มการทำนาย (Regression Line Visualization)

# แสดงกราฟเปรียบเทียบข้อมูลจริงกับเส้นทำนายของโมเดล
plt.figure(figsize=(9, 5))
plt.scatter(X_train, y_train, color='blue', alpha=0.5, label='Training Data')
plt.scatter(X_test, y_test, color='green', alpha=0.8, s=60, label='Testing Data (Actual)')
plt.plot(X_area, model.predict(X_area), color='red', linewidth=2, label=f'Regression Line: y = {slope:.3f}x + {intercept:.2f}')

plt.title("House Price Prediction using Linear Regression", fontsize=14)
plt.xlabel("Living Area (Square Meters)", fontsize=12)
plt.ylabel("Price (Million Baht)", fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
# plt.show() # Uncomment เมื่อรันใน Jupyter Notebook
print("✅ Regression Model Successfully Built and Evaluated!")

# ------------------------------------------------------------
