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
# 75% training and 25% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1) 

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()

X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

# Running SVM model

from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix,auc, roc_curve

#set start time
start_time=time()
# Create SVM classifer object
svc=SVC()
# Train Decision Tree Classifer
svc= svc.fit(X_train,y_train)
#Predict the response for test dataset
y_pred = svc.predict(X_test)
end_time=time()
runtime=end_time-start_time

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("The runtime is",runtime)
print("*******Confusion Matrix SVM ********")
print(confusion_matrix(y_test, y_pred))

#  Showing ROC Curve 


svm_fpr, svm_tpr, thresholds = roc_curve(y_test,y_pred)
auc_svm=auc(svm_fpr,svm_tpr)
plt.plot(svm_fpr,svm_tpr,linestyle='-',label='svm.SVC(auc=%0.3f)' % auc_svm)
plt.show()

# Creating Fisher Linear Discrimination


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#set start time
start_time1=time()
fld=LinearDiscriminantAnalysis()
fld.fit(X_train,y_train)
y1_pred=fld.predict(X_test)
end_time1=time()
runtime1=end_time1-start_time1

print("Accuracy:",metrics.accuracy_score(y_test, y1_pred))
print(classification_report(y_test, y1_pred))
print("The runtime is",runtime1)
print("*******Confusion Matrix FLD********")
print(confusion_matrix(y_test, y1_pred))

# Drawing ROC Curve

fld_fpr, fld_tpr, thresholds = roc_curve(y_test,y1_pred)
auc_fld=auc(fld_fpr,fld_tpr)
plt.plot(fld_fpr,fld_tpr,linestyle='-', label='LinearDiscriminantAnalysis(auc = %0.3f)'% auc_fld)
plt.show()














