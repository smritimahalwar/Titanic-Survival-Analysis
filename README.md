# Titanic Survival Analysis — Prediction & Dashboard

## 📌 Overview
This project explores the famous Titanic dataset to analyze survival patterns and build predictive models. It combines **machine learning scripts** for survival prediction with a **Power BI dashboard** for interactive visualization.

The goal was to:
- Clean and preprocess the dataset.
- Build KPIs and visuals in Power BI.
- Train at least 5 predictive models to estimate survival chances.
- Present insights in a clear, interactive way.

---

## 🎯 Problem Statement
The challenge was originally posed as:  
**“Predict which passengers survived the Titanic disaster based on features like age, sex, class, and embarkation.”**

Tasks performed:
- Handle missing values (Age, Embarked, Fare).
- Engineer features for modeling.
- Build KPIs (total passengers, survivors, non‑survivors, survival rate).
- Create visuals to show survival trends.
- Train ML models to predict survival.

---

## 📂 Repository Contents
- `data/` → Titanic dataset files  
  - `train.csv` → Training dataset used to build and evaluate models  
  - `test.csv` → Test dataset used to generate predictions  

- `scripts/` → Python scripts, predictions, and utilities  
  - `utils.py` → Helper functions for preprocessing and modeling  
  - `titanic_predictions.csv` / `.xlsx` → Prediction outputs  
  - `outputs/figures/` → Model evaluation plots (e.g., XGBoost confusion matrix)  

- `reports/` → Power BI dashboard and snapshot  
  - `Titanic_Dashboard.pbix` → Interactive dashboard  
  - `Titanic_Dashboard.pdf` → Static snapshot for quick viewing  

- `README.md` → Project explanation  

---

## ⚙️ Approach
### Data Cleaning
- Filled missing Age, Embarked, and Fare values.
- Converted categorical variables (Sex, Embarked) into numeric.

### Feature Engineering
- Created new features (e.g., family size, title extraction).
- Normalized continuous variables.

### Modeling
- Trained Logistic Regression, Decision Tree, and Random Forest models.
- Compared accuracy and performance.

### Visualization
- Built KPIs in Power BI:
  - Total Passengers (891)
  - Survivors (342)
  - Non‑Survivors (549)
  - Survival Rate (38.4%)
- Added visuals:
  - Survival by Gender
  - Survival by Passenger Class
  - Survival by Embarkation Point
  - Age distribution vs. survival
  - Fare vs. Age scatter plot
- Added slicers for Gender, Class, and Embarkation.

---

## 🔑 Key Insights
- Women had a significantly higher survival rate than men.
- First‑class passengers were more likely to survive than third‑class.
- Overall survival rate was ~38%.
- Younger passengers and those with higher fares showed different survival patterns.

---

## 🛠 Tools Used
- **Python** (pandas, scikit‑learn, matplotlib)  
- **Power BI** (DAX, dashboard design)  

---

## 🚀 How to Run
1. Clone the repository.  
2. Run Python scripts in `scripts/` to reproduce predictions.  
3. Open the `.pbix` file in Power BI Desktop to explore the dashboard.  
4. View the static snapshot in the PDF report.  

---

## 📈 Results
- Best ML model achieved ~80% accuracy.  
- Dashboard provides interactive exploration of survival patterns.  

---

## ✨ Why This Project Matters
This project demonstrates **end‑to‑end data skills**:  
- Data cleaning & preprocessing  
- Feature engineering & ML modeling  
- KPI design & dashboard storytelling  

It highlights both **technical depth** and **visual communication**, making it a strong portfolio project for resumes and LinkedIn.
