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
mpl.rcParams['axes.unicode_minus'] = False       #解决保存图像是负号'-'显示为方块的问题

#data path
data_train_file='C:/Users/Zuo/Python_Data/Titanic/train.csv'
data_test_file='C:/Users/Zuo/Python_Data/Titanic/test.csv'

data_train = pd.read_csv(data_train_file)            #read csv into dataframe format
print(data_train)

data_train.info()
dsb=data_train.describe()
print(dsb)


##EDA process
data_train.Pclass.value_counts()  #画图来看看属性和结果之间的关系


fig = plt.figure()
fig.set(alpha=0.2)   #设置图表颜色透明度


plt.subplot2grid((2,3),(0,0))                        #在一张图像里画2行3列子图
data_train.Survived.value_counts().plot(kind="bar")  #柱状图
plt.title(u"获救情况（1为获救）")                      #标题
plt.ylabel(u"人数")

plt.subplot2grid((2,3),(0,1))                        #第1行第2列的图
data_train.Pclass.value_counts().plot(kind="bar")
plt.title(u"乘客等级分布")
plt.ylabel(u"人数")

plt.subplot2grid((2,3),(0,2))
plt.scatter(data_train.Survived, data_train.Age, s=10)                             
plt.grid(b=True, which='major', axis='y')
plt.title(u"按年龄看获救分布（1为获救）")
plt.ylabel(u"年龄")                            #设定纵坐标名称
plt.axis([-0.2, 1.2, -10, 90])                 #设置x轴、y轴的取值范围

plt.subplot2grid((2,3),(1,0),colspan=2)        #两列合并
data_train.Age[data_train.Pclass == 1].plot(kind='kde')
data_train.Age[data_train.Pclass == 2].plot(kind='kde')
data_train.Age[data_train.Pclass == 3].plot(kind='kde')
plt.title(u"各等级的乘客年龄分布")
plt.xlabel(u"年龄")                                     #设置坐标轴标签
plt.ylabel(u"密度")
plt.legend((u'头等舱', u'2等舱', u'3等舱'),loc='best')  #设置图例

plt.subplot2grid((2,3),(1,2))
data_train.Embarked.value_counts().plot(kind='bar')
plt.title(u"各登船口岸上船人数")
plt.ylabel(u"人数")

plt.show()


fig = plt.figure()
fig.set(alpha=0.2)   #设置图表颜色透明度

Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
df = pd.DataFrame({u'获救':Survived_1, u'未获救':Survived_0})
df.plot(kind='bar', stacked=True)       #条形图堆叠
plt.title(u"各乘客等级的获救情况")
plt.xlabel(u"乘客等级")
plt.ylabel(u"人数")

plt.show()


from sklearn.ensemble import RandomForestRegressor


#使用RandomForestClassfier(分类器)填补缺失的Age属性
def set_missing_ages(df):    

    #把已有的数值特征取出来放到RandomForestRegressor中
    age_df = df[['Age','Fare','Parch','SibSp','Pclass']]

    #乘客分成已知年龄和未知年龄两部分
    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()

    #y即目标年龄
    y = known_age[:,0]

    #x即特征属性
    X = known_age[:,1:]

    #fit到RandomForestRegressor中
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X,y)

    #用得到的模型进行未知年龄结果预测
    predictedAges = rfr.predict(unknown_age[:,1::])

    #用得到的预测结果填补原缺失数据
    df.loc[(df.Age.isnull()), 'Age'] = predictedAges

    return df, rfr


#将Cabin按有无数据，把属性处理成Yes和No两种类型
def set_Cabin_type(df):
    df.loc[(df.Cabin.notnull()), 'Cabin'] = "Yes"
    df.loc[(df.Cabin.isnull()), 'Cabin'] = "No"

    return df


data_train, rfr = set_missing_ages(data_train)
data_train = set_Cabin_type(data_train)

#检查Age/Cabin的缺失值是否填补
print(data_train)


dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix='Cabin')
dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix='Embarked')
dummies_Sex = pd.get_dummies(data_train['Sex'], prefix='Sex')
dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix='Pclass')


df = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)

#并将下面几个原始字段从 "data_train" 中拿掉
df.drop(['Pclass','Name','Sex','Ticket','Cabin','Embarked'], axis=1, inplace=True)

print(df)

#将Age/Fare标准化到[-1,1]范围内
import sklearn.preprocessing as preprocessing
scaler = preprocessing.StandardScaler()

#data.values.reshape(-1, 1)是根据报错信息修改的，主要是因为工具包版本更新造成的
age_scale_param = scaler.fit(df['Age'].values.reshape(-1, 1))
df['Age_scaled'] = scaler.fit_transform(df['Age'].values.reshape(-1, 1), age_scale_param)

fare_scale_param = scaler.fit(df['Fare'].values.reshape(-1, 1))
df['Fare_scaled'] =scaler.fit_transform(df['Fare'].values.reshape(-1, 1), fare_scale_param)

print(df)

#逻辑回归建模
#把需要的feature字段取出来，转成numpy格式，使用scikit-learn中的LogisticRegression建模

from sklearn import linear_model

#用正则regex取出我们要的属性值
# ".*"是正则表达式中的贪婪模式，匹配任意字符0或者多次(大于等于0次)。点是任意字符，*是取0至无限长度。
train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')   #共取了15个feature字段
train_np = train_df.as_matrix()      #转成numpy格式

#y即Survived结果
y = train_np[:, 0]        #切片含义: 所有行/第1列，即Survived列

#X即特征属性值
X = train_np[:, 1:]       #切片含义: 所有行/第2列到最后一列，即Age_scaled列到Pclass_3列，共14个字段。

#fit到RandomForestRegressor中
#penalty='l1'第一个是英文字母L的小写，不是数字1；
#tol=1e-6代表科学技术法，即1乘以10的-6次方，注意这里的1不能省略，因为可能造成歧义；也可以用tol=0.000001表达。
clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)     #创建分类器对象
clf.fit(X, y)     #用训练数据拟合分类器模型

print(clf)

#------------------------------------------------------------------------------------------------
from sklearn.cross_validation import cross_val_score # K折交叉验证模块
scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
print(scores)



data_test = pd.read_csv(data_test_file)
data_test.loc[(data_test.Fare.isnull()), 'Fare'] = 0

#接着我们对test做和train中一致的特征变换
#首先用同样的RandomForestRegressor模型填上丢失的年龄
tmp_df = data_test[['Age','Fare','Parch','SibSp','Pclass']]
null_age = tmp_df[data_test.Age.isnull()].as_matrix()

#根据特征属性X预测缺失的年龄并补上
X = null_age[:, 1:]
predictedAges = rfr.predict(X)
data_test.loc[(data_test.Age.isnull()), 'Age'] = predictedAges

data_test = set_Cabin_type(data_test)
dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix='Cabin')
dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix='Embarked')
dummies_Sex = pd.get_dummies(data_test['Sex'], prefix='Sex')
dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix='Pclass')

df_test = pd.concat([data_test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
df_test['Age_scaled'] = scaler.fit_transform(df_test['Age'].values.reshape(-1, 1), age_scale_param)
df_test['Fare_scaled'] = scaler.fit_transform(df_test['Fare'].values.reshape(-1, 1), age_scale_param)

#打印预处理后的数据，检查处理效果
print(df_test)

test = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
predictions = clf.predict(test)     #用训练好的分类器去预测test数据的标签
result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(), 'Survived':predictions.astype(np.int32)})

#保存预测结果
result.to_csv("C:/Users/Zuo/Python_Data/Titanic/logistic_regression_predictions.csv", index=False)
