# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

class TurnoverClassifier:
    def __init__(self):
        self.columns = [
            'Age', 'Gender', 'MaritalStatus', 'Travelling', 'Vertical', 'Qualifications', 'EducationField', 'EmployeSatisfaction',
            'JobEngagement', 'JobLevel', 'JobSatisfaction', 'Role', 'DailyBilling', 'HourBilling', 'MonthlyBilling', 'MonthlyRate',
            'Work Experience', 'OverTime', 'PercentSalaryHike', 'Last Rating', 'RelationshipSatisfaction', 'Hours', 'StockOptionLevel',
            'TrainingTimesLastYear', 'Work&Life', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrentManager',
            'DistanceFromHome',
        ]
        self.feature_names = self.columns
        self.model = None
        self.df = None
        self.label_encoders = {}

    def load_data(self):
        try:
            self.df = pd.read_csv('ideaspiceemployeeturnoverdataset.csv')
        except FileNotFoundError:
            self.df = None

    def preprocess_data(self):
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col])
                self.label_encoders[col] = le

    def train_model(self):
        self.load_data()
        if self.df is None:
            st.error("Dataset tidak ditemukan. Pastikan file 'ideaspiceemployeeturnoverdataset.csv' tersedia.")
            return

        self.preprocess_data()
        X = self.df[self.feature_names]
        y = self.df['Turnover']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        st.subheader("🎯 Model Performance")
        st.write(f"**Accuracy:** {acc:.2f}")
        st.text("Classification Report:")
        st.text(report)

        # Simpan model dan encoders
        with open("rf_model.pkl", "wb") as f:
            pickle.dump({
                "model": model,
                "encoders": self.label_encoders
            }, f)

        self.model = model
        st.success("✅ Model berhasil dilatih dan disimpan.")

    def load_model(self):
        try:
            with open("rf_model.pkl", "rb") as f:
                data = pickle.load(f)
                self.model = data["model"]
                self.label_encoders = data["encoders"]
        except FileNotFoundError:
            self.model = None

    def predict(self, input_data):
    if self.model is None:
        self.load_model()
    if self.model is None:
        return "❌ Model belum dilatih atau file tidak ditemukan."

    # input_data sekarang dictionary, jadi bentuk langsung ke DataFrame
    input_df = pd.DataFrame([input_data])  # auto pakai key sebagai kolom

    # Encode kolom yang perlu encoding
    for col in input_df.columns:
        if col in self.label_encoders:
            le = self.label_encoders[col]
            try:
                input_df[col] = le.transform(input_df[col])
            except ValueError:
                return f"❌ Nilai input tidak dikenali untuk kolom '{col}': {input_df[col].values[0]}"

    prediction = self.model.predict(input_df.values)
    return "Yes" if prediction[0] == 1 else "No"


# Streamlit App
def main():
    st.title("💼 Employee Turnover Prediction App")
    st.sidebar.title("Menu")
    menu = ["Home", "Train Model", "Make Prediction"]
    choice = st.sidebar.selectbox("Navigation", menu)

    classifier = TurnoverClassifier()

    if choice == "Home":
        st.subheader("👋 Welcome")
        st.write("Gunakan aplikasi ini untuk melatih model dan memprediksi turnover karyawan.")

    elif choice == "Train Model":
        st.subheader("🛠️ Train the Model")
        classifier.train_model()

    elif choice == "Make Prediction":
        st.subheader("🔍 Predict Employee Turnover")

        # Input
        age = st.slider("Age", 18, 60, 30)
        gender = st.selectbox("Gender", ['Female', 'Male'])
        marital_status = st.selectbox("Marital Status", ['Single', 'Married', 'Divorced'])
        travelling = st.selectbox("Travelling", ['No', 'Mostly', 'Sometimes'])
        vertical = st.selectbox("Vertical", ['Human Resources', 'Research & Development', 'Sales'])
        qualification = st.selectbox("Qualification", [1, 2, 3, 4, 5])
        education_field = st.selectbox("Education Field", ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Other", "Human Resources"])
        satisfaction = st.slider("Employee Satisfaction", 1, 5, 3)
        engagement = st.slider("Job Engagement", 1, 5, 3)
        job_level = st.selectbox("Job Level", [1, 2, 3, 4, 5])
        job_satisfaction = st.slider("Job Satisfaction", 1, 5, 3)
        role = st.selectbox("Job Role", ["Sales Executive", "Laboratory Technician", "Research Scientist","Healthcare Representative", "Manufacturing Director",
        "Manager", "Research Director", "Sales Representative", "Human Resources"])
        daily_billing = st.number_input("Daily Billing", 100, 1500, 300)
        hour_billing = st.slider("Hourly Billing", 10, 100, 40)
        monthly_billing = st.number_input("Monthly Billing", 1000, 20000, 5000)
        monthly_rate = st.number_input("Monthly Rate", 1000, 27000, 8000)
        work_exp = st.slider("Work Experience (years)", 0, 9, 5)
        overtime = st.selectbox("OverTime", ['No', 'Yes'])
        salary_hike = st.slider("Percent Salary Hike", 10, 30, 20)
        last_rating = st.slider("Last Performance Rating", 1, 5, 3)
        rel_satisfaction = st.slider("Relationship Satisfaction", 1, 5, 3)
        hours = st.slider("Working Hours", 40, 80, 60)
        stock_option = st.selectbox("Stock Option Level", list(range(0, 11)), index=5)
        training = st.slider("Training Times Last Year", 0, 10, 3)
        work_life = st.slider("Work-Life Balance", 1, 4, 3)
        years_at_company = st.slider("Years At Company", 0, 40, 5)
        in_role = st.slider("Years In Current Role", 0, 20, 2)
        since_promo = st.slider("Years Since Last Promotion", 0, 15, 1)
        with_manager = st.slider("Years With Current Manager", 0, 20, 2)
        distance = st.slider("Distance From Home", 1, 30, 5)

        input_data = [
            age, gender, marital_status, travelling, vertical, qualification, education_field, satisfaction,
            engagement, job_level, job_satisfaction, role, daily_billing, hour_billing, monthly_billing, monthly_rate,
            work_exp, overtime, salary_hike, last_rating, rel_satisfaction, hours, stock_option,
            training, work_life, years_at_company, in_role, since_promo, with_manager, distance
        ]

        if st.button("Predict"):
            result = classifier.predict(input_data)
            st.success(f"Prediction: The employee is likely to leave? **{result}**")

if __name__ == "__main__":
    main()
