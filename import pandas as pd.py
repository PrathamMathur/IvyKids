import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.calibration import LabelEncoder

data = pd.read_csv(r"C:\Users\Ameyo\OneDrive\Desktop\churn_ds.csv")

data.head()

data.columns

data.info()

print("Missing Values:")
print(data.isnull().sum())

data.duplicated().sum()

z_scores = stats.zscore(data[['SeniorCitizen', 'tenure']])
data_no_outliers = data[(z_scores < 3) & (z_scores > -3)]
data_no_outliers['MonthlyCharges'] = data_no_outliers['tenure'].rolling(window=3).mean()
data_no_outliers['Contract'] = data_no_outliers['Contract'].str.replace('[^a-zA-Z\s]', '').str.lower()

data_encoded = pd.get_dummies(data, columns=['gender', 'Partner', 'Dependents', 'PhoneService',
                                             'MultipleLines', 'InternetService', 'OnlineSecurity',
                                             'OnlineBackup', 'DeviceProtection', 'TechSupport',
                                             'StreamingTV', 'StreamingMovies', 'Contract',
                                             'PaperlessBilling', 'PaymentMethod'], drop_first=True)
label_encoder = LabelEncoder()
data['Churn'] = label_encoder.fit_transform(data['Churn'])
print(data.describe())                                                                                            plt.figure(figsize=(6, 4))
sns.countplot(x='Churn', data=data)
plt.title('Distribution of Churn')
plt.show()                                                                                                                                numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
categorical_features = ['gender', 'Partner', 'Dependents', 'InternetService', 'Contract', 'PaymentMethod']

for feature in categorical_features:
    plt.figure(figsize=(8, 6))
    sns.countplot(x=feature, hue='Churn', data=data)
    plt.title(f'Distribution of {feature} by Churn')
    plt.show()
numeric_data = data.select_dtypes(include=['number'])
correlation_matrix = numeric_data.corr()

# Set up the matplotlib figure
plt.figure(figsize=(10, 8))

# Create a heatmap with a color map
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)

# Add a title
plt.title('Correlation Matrix Heatmap')

# Show the plot
plt.show()
