# 📊 Startup Growth & Housing Price Analysis 🚀🏡

![Startup Growth](./images/startup-growth.jpg)

## 🏠 Housing Price Classification | 📈 Startup Growth Prediction

### 🔍 Project Overview
This project explores *housing price classification* and *startup growth prediction* using machine learning.  
- *🏠 Housing Classification:* Categorizing home prices as Low, Medium, or High.  
- *📈 Startup Regression:* Predicting startup growth based on investment and valuation data.  

---

## 📂 Data Sources
1. *USA Housing Dataset* (usa_housing_kaggle.csv)
   - Features: Income, Population, Rooms, Bedrooms, House Age, etc.
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

![Distribution of House Prices](distribution%20of%20house%20prices.png)


---

## 📌 Key Insights

### 🏠 Housing Price Classification  
✅ *Top Factors Influencing Home Prices:*  
- *🏡 Area Income* → Strongest correlation with price  
- *📏 Number of Rooms* → More rooms = Higher price  
- *👥 Population Density* → Affects demand & pricing trends  

📈 *Best Model: XGBoost (72% Accuracy)*  
- Random Forest struggled, but XGBoost gave better predictions.  

#### 🔹 Feature Importance for Housing Prices  
[![Housing Feature Importance](https://raw.githubusercontent.com/crispin046/Startup_Growth_Housing_Analysis/main/House%20price%20vs%20school%20feet.png)](https://raw.githubusercontent.com/crispin046/Startup_Growth_Housing_Analysis/main/House%20price%20vs%20school%20feet.png)

---

### 📈 Startup Growth Prediction  
💡 *Investment alone does not guarantee startup success.*  
- *Funding & valuation had weak correlations with actual growth.*  
- *Market demand, innovation, and execution matter more.*  

📊 *Regression Models:*
- *Linear Regression & Random Forest both performed poorly* due to high unpredictability in startup success.  
- *External factors like industry trends and founder experience might be key missing variables.*  
  
#### 🔹 Regression Model Predictions for Startup Growth  
![Startup Growth Regression](correlation%20heatmap.png)

---

## 📂 Project Files
- *📜 Notebooks* (notebooks/)
  - Housing_EDA_Classification.ipynb – Data Exploration & Classification
  - Startup_Growth_Regression.ipynb – Regression Model
- *📜 Python Scripts* (src/)
  - housing_classification.py
  - startup_regression.py
- *📂 Data Files* (data/)

---



🔗 References

Kaggle Housing Dataset

Investment & Startup Research




---

# **📜 1. `housing_classification.py` (Housing Price Prediction)**
**📍 Location: `src/housing_classification.py`**  
This script performs classification on the USA Housing dataset.

python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("../data/usa_housing_kaggle.csv")

# Convert price to categories (Low, Medium, High)
df["Price_Category"] = pd.qcut(df["Price"], q=3, labels=["Low", "Medium", "High"])
df.drop(columns=["Price"], inplace=True)

# Encode target variable
label_encoder = LabelEncoder()
df["Price_Category"] = label_encoder.fit_transform(df["Price_Category"])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=["Price_Category"]), df["Price_Category"], test_size=0.2, random_state=42)

# Train Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predictions
y_pred = rf.predict(X_test)

# Evaluate model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Feature Importance Plot
importance = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=importance.values, y=importance.index, palette="viridis")
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Top Factors Influencing Home Prices")
plt.savefig("../images/housing_feature_importance.png")  
plt.show()


---

# **📜 2. startup_regression.py (Startup Growth Prediction)**

**📍 Location: src/startup_regression.py**

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv("../data/startup_growth_investment_data.csv").dropna()

# Train-test split
X = df.drop(columns=["Growth_Percentage"])
y = df["Growth_Percentage"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# Evaluate model
print("R² Score:", r2_score(y_test, y_pred))

# Plot results
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='black')
plt.xlabel("Actual Growth %")
plt.ylabel("Predicted Growth %")
plt.savefig("../images/startup_regression_plot.png")  
plt.show()


---
