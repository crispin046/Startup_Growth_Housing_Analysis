# 📊 Startup Growth & Housing Price Prediction

## 🏠 Housing Classification | 📈 Startup Growth Regression

### 🔍 Project Overview
This project explores *startup growth prediction* and *housing price classification* using machine learning. The goal is to:
- *Predict startup growth* based on investment and valuation data (Regression).
- *Classify homes* into price categories (Low, Medium, High) using housing attributes (Classification).

---

## 🚀 Data Used
1. *USA Housing Dataset* (usa_housing_kaggle.csv)
   - Features: Income, Area Population, Number of Rooms, House Age, etc.
   - Target: Price_Category (Low, Medium, High)
   
2. *Startup Growth & Investment Dataset* (startup_growth_investment_data.csv)
   - Features: Investment Amount, Valuation, Year Founded, etc.
   - Target: Growth Percentage

---

## 🛠 Techniques Used
- *Exploratory Data Analysis (EDA)*
- *Feature Engineering & Selection*
- *Classification (Random Forest, XGBoost) for Housing Prices*
- *Regression (Linear & Random Forest) for Startup Growth*
- *Visualizations with Seaborn & Matplotlib*

---

## 📌 Key Insights

### 🏠 Housing Price Classification
✅ *Top Factors Influencing Home Prices*  
- *🏡 Area Income:* The strongest indicator of home prices.  
- *📏 Number of Rooms:* More rooms generally mean higher prices.  
- *👥 Population Density:* Affects demand & pricing trends.  

📈 *Best Model: XGBoost (72% Accuracy)*  
- Random Forest struggled, but XGBoost provided better predictions.  

![Housing Feature Importance](images/housing_feature_importance.png)

---

### 📈 Startup Growth Prediction  
💡 *Investment alone does not guarantee startup success.*  
- *Funding & valuation had weak correlations with actual growth.*  
- *Market demand, innovation, and execution matter more.*  

📊 *Regression Models:*
- *Linear Regression & Random Forest both performed poorly* due to high unpredictability in startup success.  
- *External factors like industry trends and founder experience might be key missing variables.*  

![Startup Growth Regression](images/startup_regression_plot.png)

---

## 📂 Project Files
- *📜 Notebooks* (notebooks/)
  - EDA_and_Insights.ipynb – Data Exploration & Key Insights
  - Housing_Classification.ipynb – Housing ML Model
  - Growth_Regression.ipynb – Startup Growth ML Model
- *📜 Python Scripts* (src/)
  - housing_classification.py
  - startup_regression.py
- *📂 Data Files* (data/)

---



🔗 References

Kaggle Housing Dataset

Investment & Startup Research



---



