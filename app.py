from flask import Flask, render_template, request, flash, redirect, url_for
import pickle
import numpy as np

app = Flask(__name__)
app.secret_key = 'secret'

# Load mô hình và scaler
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('models/health_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Hàm cung cấp lời khuyên sức khỏe dựa trên BMI
def get_health_advice(bmi):
    if bmi < 18.5:
        return "Bạn bị thiếu cân. Hãy cân nhắc tăng cân thông qua chế độ ăn uống cân bằng và tập thể dục."
    elif 18.5 <= bmi < 25:
        return "Bạn có cân nặng bình thường. Hãy duy trì lối sống hiện tại."
    elif 25 <= bmi < 29.9:
        return "Bạn bị thừa cân. Hãy cân nhắc chế độ ăn uống lành mạnh và tập thể dục thường xuyên để giảm cân."
    else:
        return "Bạn bị béo phì. Hãy tìm lời khuyên từ chuyên gia chăm sóc sức khỏe để kiểm soát cân nặng."

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            # Lấy dữ liệu từ form nhập liệu
            weight = float(request.form["weight"])
            height = float(request.form["height"])
            age = int(request.form["age"])
            gender = request.form["gender"]
            activity_level = request.form["activity_level"]

            # Chuyển đổi dữ liệu thành dạng số cho mô hình
            gender = 0 if gender == "Male" else 1
            activity_level_map = {
                "Sedentary": 1.2,
                "Light activity": 1.375,
                "Moderate activity": 1.55,
                "High activity": 1.725,
                "Very high activity": 1.9
            }
            activity_level = activity_level_map[activity_level]

            # Chuẩn hóa dữ liệu
            input_data = scaler.transform([[weight, height, age, gender, activity_level]])

            # Dự đoán BMI, BMR và TDEE
            bmi, bmr, tdee = model.predict(input_data)[0]
            advice = get_health_advice(bmi)

            # Kết quả trả về
            result = {
                "bmi": round(bmi, 2),
                "bmr": round(bmr, 2),
                "tdee": round(tdee, 2),
                "advice": advice
            }
            return render_template("result.html", result=result)

        except ValueError:
            flash("Please enter valid data", "error")
            return redirect(url_for("index"))

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
