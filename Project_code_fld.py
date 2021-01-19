# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 18:12:13 2020

@author: masoodhur
"""

import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from time import  time

data=pd.read_csv("C:/Users/masoodhur/Desktop/6DA3_project/online_shoppers_intention.csv")
print(data.describe().T)
print(data.dtypes)
data.groupby("Revenue").size().plot.bar()
from sklearn import preprocessing

label_encode=preprocessing.LabelEncoder()
# Checking different types months with their corresponding values
print(data['Month'].value_counts())
# changing the values for Month because datatype is object
data["Month"]=label_encode.fit_transform(data["Month"])
print(data['Month'].value_counts())

# changing the values for VisitorType because datatype is object
data["VisitorType"]=label_encode.fit_transform(data["VisitorType"])
print(data["VisitorType"].value_counts())

# changing the values for Weekend besause data is bool
data["Weekend"]=label_encode.fit_transform(data["Weekend"])
print(data["Weekend"].value_counts())

# changing the values for  Revenue because data bool
data["Revenue"]=label_encode.fit_transform(data["Revenue"])
print(data["Revenue"].value_counts())


# Checking for any null values
print(data.isnull().sum().sum())

df=data.copy()

# outlier detection and removal
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
print(IQR)

print((df < (Q1 - 1.5 * IQR))|(df > (Q3 + 1.5 * IQR)).sum().sum())

df_out = df[~((df <(Q1 - 1.5 * IQR)) | (df>(Q3 + 1.5 * IQR))).any(axis=1)]
# Total outlier cleaned
print(df_out.shape)

df_cleaned=df[((df <(Q1 - 1.5 * IQR)) | (df>(Q3 + 1.5 * IQR))).any(axis=1)]

# Cheeking some of the variables for clean dataset

plt.boxplot(df_cleaned["Administrative"],vert= False,meanline=True,showmeans=True)
plt.show()

plt.boxplot(df_cleaned["Administrative_Duration"],vert= False,meanline=True,showmeans=True)
plt.show()

plt.boxplot(df_cleaned["Informational"],vert= False,meanline=True,showmeans=True)
plt.show()

print(df_cleaned.columns)

from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest

X = df_cleaned.drop('Revenue',axis=1)
y = df_cleaned['Revenue']


test = SelectKBest(score_func=chi2, k=10)

fit = test.fit(X, y)

#Select the Top 10 features
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(10,'Score'))  #print 10 best features
# Creating dataframe using best 10 dataset
features_name=['ProductRelated_Duration','PageValues','Administrative_Duration','Informational_Duration',
               'ProductRelated','Administrative','Informational','Month','SpecialDay','TrafficType']


X=df_cleaned[features_name]
y=df_cleaned["Revenue"]

from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics
# Split the dataset into 75% training and 25% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1) 

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()

X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)


# Creating Fisher Linear Discrimination


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report, confusion_matrix,auc, roc_curve

# Create FLD classifier object
fld=LinearDiscriminantAnalysis()
#set start train time
start_time_train=time()
# Train  FLD Classifier
fld.fit(X_train,y_train)
# End time for train
end_time_train=time()
train_time=end_time_train-start_time_train
#set start test time
start_test_time=time()
#Predict the response for test dataset
y1_pred=fld.predict(X_test)
# Take end time for  test
end__test_time=time()
test_time=end__test_time-start_test_time

print("Accuracy rate for FLD:",metrics.accuracy_score(y_test, y1_pred))
print(classification_report(y_test, y1_pred))
print("*******Confusion Matrix FLD********")
print(confusion_matrix(y_test, y1_pred))
print("The train time for FLD",train_time)
print("The test time for FLD",test_time)
# Drawing ROC Curve

fld_fpr, fld_tpr, thresholds = roc_curve(y_test,y1_pred)
auc_fld=auc(fld_fpr,fld_tpr)
plt.plot(fld_fpr,fld_tpr,linestyle='-', label='LinearDiscriminantAnalysis(auc = %0.3f)'% auc_fld)
plt.plot([0, 1],[0, 1])
plt.show()














