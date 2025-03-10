
# Customer Churn Prediction Using Artificial Neural Network (ANN)

## Overview
Customer churn prediction helps businesses understand why customers leave and take proactive steps to retain them. This project focuses on predicting customer churn in the telecom industry using a deep learning model (Artificial Neural Network). 
---

## Dataset
- **Dataset Name:** `customer_churn.csv`
- **Rows:** 7043  
- **Columns:** 21  
- **Target Variable:** `Churn` (Yes/No)

---

## Technologies Used
- Python  
- Pandas  
- NumPy  
- Matplotlib  
- Scikit-learn  
- TensorFlow/Keras  

---

## Steps Involved

### Step 1: Data Preprocessing
- Dropped the `customerID` column as it is not relevant for prediction.  
- Converted `TotalCharges` from object to float and handled missing values.  
- Encoded categorical variables such as gender, contract type, and payment method.  

```python
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# Load dataset
df = pd.read_csv("customer_churn.csv")

# Drop customer ID
df.drop('customerID', axis=1, inplace=True)

# Convert TotalCharges to numeric (handling spaces)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Drop rows with missing TotalCharges values
df.dropna(subset=['TotalCharges'], inplace=True)

# Encode categorical variables
df = pd.get_dummies(df, drop_first=True)

# Splitting dataset
from sklearn.model_selection import train_test_split
X = df.drop("Churn_Yes", axis=1)
y = df["Churn_Yes"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

### Step 2: Model Building (ANN)
- Built an Artificial Neural Network (ANN) using TensorFlow/Keras.  
- Model architecture:
  - **Input Layer:** 26 neurons (matching the number of features).  
  - **Hidden Layers:** 2 layers (26 and 15 neurons, ReLU activation).  
  - **Output Layer:** 1 neuron (sigmoid activation for binary classification).  

```python
import tensorflow as tf
from tensorflow import keras

# Define ANN model
model = keras.Sequential([
    keras.layers.Dense(26, input_shape=(26,), activation='relu'),
    keras.layers.Dense(15, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=100, batch_size=32)
```

---

### Step 3: Model Evaluation
- Used confusion matrix and classification report to measure performance.  
- Evaluated using precision, recall, and F1-score.

```python
from sklearn.metrics import confusion_matrix, classification_report

# Predictions
y_pred = (model.predict(X_test) > 0.5).astype("int32")

# Model evaluation
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
```

---

## Results
- Achieved high accuracy with a good balance between precision and recall.  
- Evaluated using F1-score to ensure effective churn prediction.  

---

## How to Run the Project
### 1. Clone the Repository
```bash
git clone https://github.com/your-username/customer-churn-prediction.git
cd customer-churn-prediction
```
### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
### 3. Run the Jupyter Notebook
```bash
jupyter notebook
```
Open `customer_churn_prediction.ipynb` and run the cells.

---

## Future Improvements
- Hyperparameter tuning for better accuracy.  
- Experimenting with other machine learning models such as Random Forest and XGBoost.  
- Feature engineering for better insights.  

---

## Contributors
**Bhavagna Shreya Bnadaru** -(bbandar5@asu.edu) 

---

If you find this project helpful, consider giving it a star on GitHub.  
```
