import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

# Bước 1: Load dữ liệu
data = pd.read_csv('data/expanded_health_data_fully_english (1).csv')

# Bước 2: Chuyển đổi các cột dạng chuỗi thành số
data['gender'] = data['gender'].map({"Male": 0, "Female": 1})
activity_map = {
    "Sedentary": 1.2,
    "Light activity": 1.375,
    "Moderate activity": 1.55,
    "High activity": 1.725,
    "Very high activity": 1.9
}
data['activity_level'] = data['activity_level'].map(activity_map)

# Bước 3: Tính toán các chỉ số BMI, BMR và TDEE cho dữ liệu
data['height_m'] = data['height'] / 100  # Chuyển đổi chiều cao từ cm sang mét
data['bmi'] = data['weight'] / (data['height_m'] ** 2)

# Tính BMR dựa trên công thức Harris-Benedict
data['bmr'] = np.where(
    data['gender'] == 0,
    88.362 + (13.397 * data['weight']) + (4.799 * data['height']) - (5.677 * data['age']),
    447.593 + (9.247 * data['weight']) + (3.098 * data['height']) - (4.330 * data['age'])
)

# Tính TDEE dựa trên BMR và mức độ hoạt động
data['tdee'] = data['bmr'] * data['activity_level']

# Bước 4: Xác định các biến đầu vào (X) và đầu ra (y)
X = data[['weight', 'height', 'age', 'gender', 'activity_level']]
y = data[['bmi', 'bmr', 'tdee']]

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Bước 5: Tách dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Bước 6: Tạo và huấn luyện mô hình hồi quy
model = LinearRegression()
model.fit(X_train, y_train)

# Đánh giá mô hình
score = model.score(X_test, y_test)
print(f"Model R^2 Score: {score}")

# Bước 7: Lưu scaler và mô hình
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('models/health_model.pkl', 'wb') as f:
    pickle.dump(model, f)
