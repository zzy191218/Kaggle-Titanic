# -*- coding: utf-8 -*-
"""
Created on Wed May 16 00:55:36 2018

@author: Zuo
"""

import pandas as pd              
import numpy as np                    
from pandas import Series, DataFrame

import matplotlib.pyplot as plt

from pylab import mpl                            #show Chinese Charactor
mpl.rcParams['font.sans-serif'] = ['FangSong']   #setting Deflault Chinese Format
mpl.rcParams['axes.unicode_minus'] = False       #fix the problem when you save the graph, the '-' shows as a box

#data path
data_train_file='C:/Users/Zuo/Python_Data/Titanic/train.csv'
data_test_file='C:/Users/Zuo/Python_Data/Titanic/test.csv'

data_train = pd.read_csv(data_train_file)            #read csv into dataframe format
print(data_train)

data_train.info()
dsb=data_train.describe()
print(dsb)


##----------------------------------------------EDA process----------------------------------------------------
data_train.Pclass.value_counts()  


fig = plt.figure()
fig.set(alpha=0.2)   #Set the default color transparency


plt.subplot2grid((2,3),(0,0))                        #2 row and 3 column subplot
data_train.Survived.value_counts().plot(kind="bar")  #bar chart
plt.title(u"获救情况（1为获救）")                      #title
plt.ylabel(u"人数")

plt.subplot2grid((2,3),(0,1))                        #the subplot on the first row and second column
data_train.Pclass.value_counts().plot(kind="bar")
plt.title(u"乘客等级分布")
plt.ylabel(u"人数")

plt.subplot2grid((2,3),(0,2))
plt.scatter(data_train.Survived, data_train.Age, s=10)                             
plt.grid(b=True, which='major', axis='y')
plt.title(u"按年龄看获救分布（1为获救）")
plt.ylabel(u"年龄")                            
plt.axis([-0.2, 1.2, -10, 90])                 

plt.subplot2grid((2,3),(1,0),colspan=2)        
data_train.Age[data_train.Pclass == 1].plot(kind='kde')
data_train.Age[data_train.Pclass == 2].plot(kind='kde')
data_train.Age[data_train.Pclass == 3].plot(kind='kde')
plt.title(u"各等级的乘客年龄分布")
plt.xlabel(u"年龄")                                     
plt.ylabel(u"密度")
plt.legend((u'头等舱', u'2等舱', u'3等舱'),loc='best')  

plt.subplot2grid((2,3),(1,2))
data_train.Embarked.value_counts().plot(kind='bar')
plt.title(u"各登船口岸上船人数")
plt.ylabel(u"人数")

plt.show()


fig = plt.figure()
fig.set(alpha=0.2)   

Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
df = pd.DataFrame({u'获救':Survived_1, u'未获救':Survived_0})
df.plot(kind='bar', stacked=True)       
plt.title(u"各乘客等级的获救情况")
plt.xlabel(u"乘客等级")
plt.ylabel(u"人数")

plt.show()

##-------------------------------------------Data Cleaning and Feature Selection----------------------------------------
from sklearn.ensemble import RandomForestRegressor

#define a function to fix missing values for Age
def set_missing_ages(df):    
    age_df = df[['Age','Fare','Parch','SibSp','Pclass']]

    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()

    y = known_age[:,0]
    X = known_age[:,1:]

    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X,y)

    predictedAges = rfr.predict(unknown_age[:,1::])

    df.loc[(df.Age.isnull()), 'Age'] = predictedAges

    return df, rfr


#define a function to fix missing values for Cabin
def set_Cabin_type(df):
    df.loc[(df.Cabin.notnull()), 'Cabin'] = "Yes"
    df.loc[(df.Cabin.isnull()), 'Cabin'] = "No"

    return df


data_train, rfr = set_missing_ages(data_train)
data_train = set_Cabin_type(data_train)

#check whether the missing value for Age and Cabin are filled
print(data_train)


dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix='Cabin')
dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix='Embarked')
dummies_Sex = pd.get_dummies(data_train['Sex'], prefix='Sex')
dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix='Pclass')


df = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)

#Remove several original columns
df.drop(['Pclass','Name','Sex','Ticket','Cabin','Embarked'], axis=1, inplace=True)

print(df)

#transfrom the value into scale[-1,1], for Age and Fare
import sklearn.preprocessing as preprocessing
scaler = preprocessing.StandardScaler()


age_scale_param = scaler.fit(df['Age'].values.reshape(-1, 1))
df['Age_scaled'] = scaler.fit_transform(df['Age'].values.reshape(-1, 1), age_scale_param)

fare_scale_param = scaler.fit(df['Fare'].values.reshape(-1, 1))
df['Fare_scaled'] =scaler.fit_transform(df['Fare'].values.reshape(-1, 1), fare_scale_param)

print(df)


#---------------------------------------------Build Logistic regression model-----------------------------------------------------
#Build Logistic regression model

from sklearn import linear_model

#Using regex to select the column we want
# ".*"是正则表达式中的贪婪模式，匹配任意字符0或者多次(大于等于0次)。点是任意字符，*是取0至无限长度。
train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')   #select 15 features
train_np = train_df.as_matrix()      #change to Numpy format


y = train_np[:, 0]        #Slice data label

#X即特征属性值
X = train_np[:, 1:]       #Slice 14 data features

#fit到RandomForestRegressor中
#penalty='l1' the first is Lowercase of 'L' instead of number 1
#tol=1e-6 is scitific format
clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)     #create a classifier
clf.fit(X, y)     #train this classifier

print(clf)


#----------------------------------------We do the same thing to Test Data---------------------------------------------------

data_test = pd.read_csv(data_test_file)
data_test.loc[(data_test.Fare.isnull()), 'Fare'] = 0


tmp_df = data_test[['Age','Fare','Parch','SibSp','Pclass']]
null_age = tmp_df[data_test.Age.isnull()].as_matrix()

#fill the missing Age value
X = null_age[:, 1:]
predictedAges = rfr.predict(X)
data_test.loc[(data_test.Age.isnull()), 'Age'] = predictedAges

#fill the missing Cabin value
data_test = set_Cabin_type(data_test)
dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix='Cabin')
dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix='Embarked')
dummies_Sex = pd.get_dummies(data_test['Sex'], prefix='Sex')
dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix='Pclass')

df_test = pd.concat([data_test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
df_test['Age_scaled'] = scaler.fit_transform(df_test['Age'].values.reshape(-1, 1), age_scale_param)
df_test['Fare_scaled'] = scaler.fit_transform(df_test['Fare'].values.reshape(-1, 1), age_scale_param)


print(df_test)

test = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
predictions = clf.predict(test)     #predict the label for the test data set
result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(), 'Survived':predictions.astype(np.int32)})

#write to a CSV file
result.to_csv("C:/Users/Zuo/Python_Data/Titanic/logistic_regression_predictions.csv", index=False)
