#!/usr/bin/env python
# coding: utf-8

# ## Loan Rejection or Approval Status Prediction

# In[1]:


import pandas as pd

# Load the dataset
file_path = 'loan_data_1.csv'
loan_data = pd.read_csv(file_path)

# Display the first few rows of the dataset to understand its structure and content
loan_data.head()


# In[2]:


# Check for missing values in the dataset
loan_data.isnull().sum()


# In[3]:


# Drop the redundant column
loan_data = loan_data.drop(columns=['Unnamed: 0'])

# Fill missing values for categorical variables with the mode
categorical_columns = ['Gender', 'Dependents', 'Education', 'Self_Employed']
for column in categorical_columns:
    loan_data[column] = loan_data[column].fillna(loan_data[column].mode()[0])

# Fill missing values for numerical variables with the median
numerical_columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
for column in numerical_columns:
    loan_data[column] = loan_data[column].fillna(loan_data[column].median())

# Check if all missing values are filled
loan_data.isnull().sum()


# In[4]:


# Encode categorical variables using one-hot encoding
loan_data_encoded = pd.get_dummies(loan_data, columns=categorical_columns + ['Married', 'Property_Area'], drop_first=True)

# Display the first few rows of the encoded dataset to verify the changes
loan_data_encoded.head()


# In[5]:


# Drop the 'Loan_ID' column as it is not useful for prediction
loan_data_final = loan_data_encoded.drop(columns=['Loan_ID'])

# Separate the features and the target variable
X = loan_data_final.drop('Loan_Status', axis=1)  # Features
y = loan_data_final['Loan_Status']  # Target variable

# Encode the target variable to numeric
y = y.map({'Y': 1, 'N': 0})

X.head(), y.head()


# In[6]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the Random Forest classifier
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

accuracy, classification_rep


# In[ ]:





# Building a machine learning model to predict whether loan applications should be approved or rejected is a valuable objective, especially for financial institutions aiming to automate and optimize their loan decision processes. Here are some insights and feedback on this goal:
# 
# ### Insights
# 
# 1. **Data Quality and Completeness**: The quality and completeness of the data are crucial. In the provided dataset, there were missing values in several columns that needed to be addressed. Ensuring high-quality data can significantly improve model accuracy.
# 
# 2. **Feature Selection and Engineering**: The choice of features included in the model can greatly affect its performance. Some features might have a stronger impact on loan approval decisions than others. Additionally, creating new features (feature engineering) can sometimes uncover patterns that are not immediately apparent.
# 
# 3. **Model Interpretability**: For financial decisions, such as loan approvals, it's important not only to have a model that predicts accurately but also one that stakeholders can understand and trust. Models that provide interpretable decisions can be more easily validated and accepted.
# 
# 4. **Bias and Fairness**: It's critical to ensure that the model does not inadvertently discriminate against certain groups of applicants. Regular checks for bias and fairness in the model's predictions are essential to maintain ethical standards and comply with regulatory requirements.
# 
# ### Feedback
# 
# 1. **Model Selection**: While Random Forest is a strong choice due to its robustness and ability to handle different types of data, exploring other models like Gradient Boosting, SVM, or even neural networks might yield better performance or insights.
# 
# 2. **Cross-Validation**: To ensure that the model performs well on unseen data, implementing cross-validation during the training phase can help assess its true predictive power and reduce the likelihood of overfitting.
# 
# 3. **Hyperparameter Tuning**: Fine-tuning the model's hyperparameters can lead to better performance. Techniques like grid search or random search can be employed to find the optimal settings.
# 
# 4. **Performance Metrics**: While accuracy is a good starting point, considering other metrics like ROC-AUC, precision-recall curves, and confusion matrices can provide a more nuanced view of the model's performance, especially in imbalanced datasets.
# 
# 5. **Deployment Considerations**: Planning for model deployment, including how it will be integrated into existing systems, how often it will be retrained, and how it will handle new or evolving types of loan applications, is crucial for long-term success.
# 
# In summary, while the initial model building is promising, continuous improvement, monitoring, and validation are key to ensuring that the model remains effective and fair over time.

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




