import pandas as pd
import seaborn as sns
df = pd.read_csv("BankNote_Authentication.csv")
print(df.head())

from sklearn.model_selection import train_test_split 
variance = df["variance"]
Class = df["class"]

variance_train, variance_test, Class_train, Class_test = train_test_split(variance, Class, test_size = 0.25, random_state = 0)
from sklearn.linear_model import LogisticRegression
import numpy as np

X = np.reshape(variance_train.ravel(), (len(variance_train), 1))
Y = np.reshape(Class_train.ravel(), (len(Class_train), 1))

classifier = LogisticRegression(random_state = 0) 
classifier.fit(X, Y)

X_test = np.reshape(variance_test.ravel(), (len(variance_test), 1))
Y_test = np.reshape(Class_test.ravel(), (len(Class_test), 1))

Class_prediction = classifier.predict(X_test)

predicted_values = []
for i in Class_prediction:
  if i == 0:
    predicted_values.append("Authorized")
  else:
    predicted_values.append("Forged")

actual_values = []
for i in Y_test.ravel():
  if i == 0:
    actual_values.append("Authorized")
  else:
    actual_values.append("Forged")
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt  

labels = ["Forged", "Authorised"]
cm = confusion_matrix(actual_values, predicted_values, labels)

ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax)

ax.set_xlabel('Predicted')
ax.set_ylabel('Actual') 
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(labels); ax.yaxis.set_ticklabels(labels)