import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# 1. Data Preprocessing
df = pd.read_csv('Churn_Modelling.csv') # Corrected filename
df.head()
df.info()
df.describe()
df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
df.head()
df.info()
df.describe()
# Check for missing values
df.isnull().sum()

# 2. Feature Engineering
df = pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=True)

# Separate features x and target y
X = df.drop('Exited', axis=1)
y = df['Exited']
#data visualization
#  donot  chart for target variable 
sns.countplot(x='Exited', data=df)
plt.show()
# 3. Feature Scaling 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X) 

# 4. Data Splitting
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train.shape, X_test.shape

# 5. Model Training
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
model = LogisticRegression(max_iter=1000) 
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy_score(y_test, y_pred)
# 6. Model Evaluation
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
comparison.head()
#eveluation f1 , recall, auc
from sklearn.metrics import f1_score, recall_score, roc_auc_score
print(f1_score(y_test, y_pred))
print(recall_score(y_test, y_pred))
print(roc_auc_score(y_test, y_pred))

# Confusion Matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))

#x and y axis  compare the performance of the model using data visualization