# Stroke-Predicting-with-Machine-Learning-models

**Data Source Introduction:** 
This dataset is based on a total population of 5110 people which includes 2995 females and 2115 males. The dataset for this study is extracted from https://www.kaggle.com, it predicts whether a patient is likely to get stroke based on the following attribute : ID, Gender, Age, Hypertension, Heart_disease, Ever_married, Work_type, Residence, Avg_glucose, BMI, Smoking_status, and stroke (Response Variable).
Attribute Information
1) id: unique identifier
2) gender: "Male", "Female" or "Other"
3) age: age of the patient
4) hypertension: 0 if the patient doesn't have hypertension, 1 if the patient has hypertension
5) heart_disease: 0 if the patient doesn't have any heart diseases, 1 if the patient has a heart disease
6) ever_married: "No" or "Yes"
7) work_type: "children", "Govt_jov", "Never_worked", "Private" or "Self-employed"
8) Residence_type: "Rural" or "Urban"
9) avg_glucose_level: average glucose level in blood
10) bmi: body mass index
11) smoking_status: "formerly smoked", "never smoked", "smokes" or "Unknown"*
12) stroke: 1 if the patient had a stroke or 0 if not
*Note: "Unknown" in smoking_status means that the information is unavailable for this patient

**Models Used:**
The target variable is ‘stroke’. Here 5 different models is used in order to determine which ones produce the most accurate,  reliable and repeatable outcomes. The models we used are named below;
●	PCA Analysis 
●	Decision Tree
●	Neural Network
●	Random Forest
●	Cluster Analysis
The codes are attached in the repository. While validating the data with the models decision tree showed 95.2% accuracy, Neural net shoed an accuracy of 95%, and Random forest had an accuracy of 93%.
