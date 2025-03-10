# Customer_Churn_Prediction
Customer Churn Prediction Using Artificial Neural Network (ANN)
Overview
Customer churn prediction helps businesses understand why customers leave and take proactive steps to retain them. This project focuses on predicting customer churn in the telecom industry using a deep learning model (Artificial Neural Network). The model is evaluated using precision, recall, and F1-score.

Dataset
The dataset contains customer information, such as tenure, services subscribed, contract type, and payment method, along with the target variable Churn (Yes/No).

Dataset Name: customer_churn.csv
Rows: 7043
Columns: 21
Technologies Used
Python
Pandas
NumPy
Matplotlib
Scikit-learn
TensorFlow/Keras
Steps Involved
1. Data Preprocessing
Dropped customerID (not useful for prediction).
Converted TotalCharges from object to float (some missing values were found and handled).
Handled categorical variables (e.g., gender, contract type, and payment method).
2. Model Building
Built an Artificial Neural Network (ANN) with TensorFlow/Keras.
Architecture:
Input Layer: 26 neurons (matching the number of features).
Hidden Layers: 2 layers (26 and 15 neurons, ReLU activation).
Output Layer: 1 neuron (sigmoid activation for binary classification).
python
Copy
Edit
import tensorflow as tf
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Dense(26, input_shape=(26,), activation='relu'),
    keras.layers.Dense(15, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100)
3. Model Evaluation
Used confusion matrix and classification report to measure performance.
python
Copy
Edit
from sklearn.metrics import confusion_matrix, classification_report

print(classification_report(y_test, y_pred))
Results
The model achieved high accuracy, with a good balance between precision and recall.
The F1-score was used to evaluate performance, ensuring the model is effective in predicting churn.
How to Run the Project
Clone the Repository

bash
Copy
Edit
git clone https://github.com/your-username/customer-churn-prediction.git
cd customer-churn-prediction
Install Dependencies

bash
Copy
Edit
pip install -r requirements.txt
Run the Jupyter Notebook

bash
Copy
Edit
jupyter notebook
Open customer_churn_prediction.ipynb and run the cells.

Future Improvements
Hyperparameter tuning for better accuracy.
Experimenting with other ML models (Random Forest, XGBoost).
Feature engineering for better insights.
Contributors
Your Name (your-email@example.com)
