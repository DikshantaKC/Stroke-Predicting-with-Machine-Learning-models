
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
data = pd.read_excel(&#39;Stroke Data.xlsx&#39;)
data.fillna(data.mean(), inplace=True)
#data =
data.drop([&#39;age&#39;,&#39;avg_glucose_level&#39;,&#39;bmi&#39;,&#39;age&#39;,&#39;heart_disease&#39;,&#39;ever_married&#39;,&#39;work_type&#39;,&#39;smokin
g_status&#39;,&#39;hypertension&#39;], axis=1)
labelEncoder = LabelEncoder()
labelEncoder.fit(data[&#39;gender&#39;])
data[&#39;gender&#39;] = labelEncoder.transform(data[&#39;gender&#39;])
labelEncoder = LabelEncoder()
labelEncoder.fit(data[&#39;ever_married&#39;])
data[&#39;ever_married&#39;] = labelEncoder.transform(data[&#39;ever_married&#39;])
labelEncoder = LabelEncoder()
labelEncoder.fit(data[&#39;work_type&#39;])
data[&#39;work_type&#39;] = labelEncoder.transform(data[&#39;work_type&#39;])
labelEncoder = LabelEncoder()
labelEncoder.fit(data[&#39;Residence_type&#39;])
data[&#39;Residence_type&#39;] = labelEncoder.transform(data[&#39;Residence_type&#39;])
labelEncoder = LabelEncoder()
labelEncoder.fit(data[&#39;smoking_status&#39;])
data[&#39;smoking_status&#39;] = labelEncoder.transform(data[&#39;smoking_status&#39;])
predictors = [&#39;age&#39;,&#39;gender&#39;,&#39;Residence_type&#39;]
X = data[predictors]
y = data[&#39;stroke&#39;]

24

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.4, random_state=0)
estimator = DecisionTreeClassifier()
estimator.fit(X_train, y_train)
print(&#39;depth&#39;,&#39;gini&#39;,&#39;entropy&#39;)
for i in range(1,30):
dtree=DecisionTreeClassifier(criterion=&#39;gini&#39;, max_depth=i)
dtree.fit(X_train, y_train)
pred = dtree.predict(X_test)
gini_score = accuracy_score(y_test, pred)
dtree=DecisionTreeClassifier(criterion=&#39;entropy&#39;, max_depth=i)
dtree.fit(X_train, y_train)
pred = dtree.predict(X_test)
entropy_score = accuracy_score(y_test, pred)
print(f&#39;{i:&lt;6}{round(gini_score,3):&lt;6}{round(entropy_score,3)}&#39;)
print()
print()
dtree = DecisionTreeClassifier(criterion=&#39;entropy&#39;, max_depth=(3))
dtree.fit(X_train, y_train)
pred = dtree.predict(X_test)
print(&quot;Accuracy:&quot;, round(accuracy_score(y_test, pred), 3))
confusionMatrix = pd.DataFrame(
confusion_matrix(y_test, pred),
columns=[&#39;Predicted No Stroke&#39;,&#39;Predicted Stroke&#39;],
index=[&#39;True No Stroke&#39;,&#39;True Stroke&#39;]
)
print(confusionMatrix)
print()
importances=
pd.DataFrame({&#39;predictor&#39;:X_train.columns,&#39;importance&#39;:np.round(dtree.feature_importances_,3)
})
importances= importances.sort_values(&#39;importance&#39;,ascending=(False))
print(importances)
print()

25

plt.figure()
plot_tree(dtree, filled=True, feature_names=predictors, class_names=[&#39;No&#39;,&#39;Yes&#39;])
plt.savefig(&#39;treePlot.pdf&#39;)
plt.show()