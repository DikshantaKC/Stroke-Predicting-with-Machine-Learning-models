
"""
Random Forest Classifier
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler


data = pd.read_excel('Stroke Data.xlsx')
data.fillna(data.mean(), inplace=True)

#data = data.drop(['gender','Residence_type','avg_glucose_level','bmi','ever_married','work_type','smoking_status','hypertension'], axis=1)

labelEncoder = LabelEncoder()
labelEncoder.fit(data['gender'])
data['gender'] = labelEncoder.transform(data['gender'])
labelEncoder = LabelEncoder()
labelEncoder.fit(data['ever_married'])
data['ever_married'] = labelEncoder.transform(data['ever_married'])
labelEncoder = LabelEncoder()
labelEncoder.fit(data['work_type'])
data['work_type'] = labelEncoder.transform(data['work_type'])
labelEncoder = LabelEncoder()
labelEncoder.fit(data['Residence_type'])
data['Residence_type'] = labelEncoder.transform(data['Residence_type'])
labelEncoder = LabelEncoder()
labelEncoder.fit(data['smoking_status'])
data['smoking_status'] = labelEncoder.transform(data['smoking_status'])

# Predictors and Target Variable
predictors = ['age','heart_disease']
X=data[predictors]
y=data['stroke']

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0) # 70% training and 30% test

#Normalize your data Here!!
scaler = StandardScaler()

# Fit only to the training data
scaler.fit(X_train)
StandardScaler(copy=True, with_mean=True, with_std=True)


#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)

print('depth','gini','entropy')
for i in range(1,31):
    dtree=RandomForestClassifier(criterion='gini', max_depth=i)
    dtree.fit(X_train, y_train)
    pred = dtree.predict(X_test)
    gini_score = accuracy_score(y_test, pred)
    
    dtree=RandomForestClassifier(criterion='entropy', max_depth=i)
    dtree.fit(X_train, y_train)
    pred = dtree.predict(X_test)
    entropy_score = accuracy_score(y_test, pred)
    
    print(f'{i:<6}{round(gini_score,3):<6}{round(entropy_score,3)}')

print()
print()

dtree = RandomForestClassifier(criterion='entropy', max_depth=(3))
dtree.fit(X_train, y_train)

pred = dtree.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, pred), 3))

confusionMatrix = pd.DataFrame(
    confusion_matrix(y_test, pred),
    columns=['Predicted No Stroke','Predicted Stroke'],
    index=['True No Stroke','True Stroke']
    )

print(confusionMatrix)
print()

importances= pd.DataFrame({'predictor':X_train.columns,'importance':np.round(dtree.feature_importances_,3)})
importances= importances.sort_values('importance',ascending=(False))
print(importances)
print()





