# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats

# Load the dataset
housing_data = pd.read_csv("/kaggle/input/income-restricted/income-restricted-inventory-2021.csv")

# Display dataset structure
housing_data.info()

# Summary statistics of the dataset
print(housing_data.describe())

# Checking for missing values
missing_values = housing_data.isna().sum()
print("Missing values per column:\n", missing_values)

# Drop rows with missing values in critical columns
housing_data_clean = housing_data.dropna(subset=['MarketRent', 'RentUnits', 'Total Income-Restricted'])

# Scatter plot of Total Income Restricted Units vs Total Project Units
plt.figure(figsize=(8, 6))
sns.scatterplot(data=housing_data_clean, x="TtlProjUnits", y="Total Income-Restricted", hue="Tenure")
plt.title("Total Income-Restricted Units vs Total Project Units")
plt.xlabel("Total Project Units")
plt.ylabel("Total Income-Restricted Units")
plt.show()

# Linear regression for Total Project Units vs Total Income-Restricted Units
X = sm.add_constant(housing_data_clean["TtlProjUnits"])  # Add constant
y = housing_data_clean["Total Income-Restricted"]
lin_reg = sm.OLS(y, X).fit()
print(lin_reg.summary())

# Scatter plot of Market Rent vs Total Rental Units
plt.figure(figsize=(8, 6))
sns.scatterplot(data=housing_data_clean, x="RentUnits", y="MarketRent", hue="Neighborhood")
plt.title("Market Rent vs Total Rental Units")
plt.xlabel("Total Rental Units")
plt.ylabel("Market Rent")
plt.show()

# Linear regression for Rent Units vs Market Rent
X2 = sm.add_constant(housing_data_clean["RentUnits"])
y2 = housing_data_clean["MarketRent"]
lin_reg2 = sm.OLS(y2, X2).fit()
print(lin_reg2.summary())

# 1. Relationship between income-restricted units and market rent
plt.figure(figsize=(8, 6))
sns.scatterplot(data=housing_data_clean, x="Total Income-Restricted", y="MarketRent")
sns.regplot(data=housing_data_clean, x="Total Income-Restricted", y="MarketRent", scatter=False, color='red')
plt.title("Income-Restricted Units vs Market Rent")
plt.xlabel("Total Income-Restricted Units")
plt.ylabel("Market Rent")
plt.show()

# Linear regression: Income-Restricted Units vs Market Rent
X_fit = sm.add_constant(housing_data_clean["Total Income-Restricted"])
y_fit = housing_data_clean["MarketRent"]
fit = sm.OLS(y_fit, X_fit).fit()
print(fit.summary())

# 2. Relationship between income-restricted units and tenure (public or private)
contingency_table = pd.crosstab(housing_data_clean['Total Income-Restricted'], housing_data_clean['Tenure'])
print("Contingency Table:\n", contingency_table)

# Chi-square test for independence
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
print(f"Chi-Square Test: chi2 = {chi2}, p-value = {p}")

# 3. Effect of total project units on the ratio of income-restricted to market rate units
housing_data_clean['Ratio'] = housing_data_clean['Total Income-Restricted'] / housing_data_clean['TtlMarket']

# Correlation analysis: Total Project Units vs Ratio
correlation = housing_data_clean['TtlProjUnits'].corr(housing_data_clean['Ratio'])
print(f"Correlation between Total Project Units and Ratio: {correlation}")

# Linear regression: Total Project Units vs Ratio
X_ratio = sm.add_constant(housing_data_clean["TtlProjUnits"])
y_ratio = housing_data_clean["Ratio"]
model = sm.OLS(y_ratio, X_ratio).fit()
print(model.summary())
