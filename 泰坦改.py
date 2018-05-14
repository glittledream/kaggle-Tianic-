import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import re
data = pd.read_csv("/home/gch/桌面/train.csv")
#名字名称
train_data = data
train_data["Title"] = train_data["Name"].map(lambda  x: re.compile(",(.*?)\.").findall(x)[0])
train_data["Title"] = train_data["Title"].map(str.strip)
train_data["FareBins"]=pd.qcut(train_data["Fare"],5)
df['Title'][df.Title=='Jonkheer'] = 'Master'
df['Title'][df.Title.isin(['Ms','Mlle'])] = 'Miss'
df['Title'][df.Title.isin(['Mme','Dona', 'Lady', 'the Countess'])] = 'Mrs'
df['Title'][df.Title.isin(['Capt', 'Don', 'Major', 'Col', 'Sir'])] = 'Mr'
df['Title'][df.Title.isin(['Dr','Rev'])] = 'DrAndRev'
train_data["Title_id"] = pd.factorize(train_data.Title)[0]



#家族
train_data["Family"] = train_data["SibSp"]+train_data["Parch"]
train_data["Family"][train_data.Family == 0] = 0
train_data["Family"][train_data.Family == 1] = 1



#票价
train_data["FareBins"] = pd.qcut(train_data["Fare"],5)
train_data["Fareclass"]= pd.factorize(train_data.FareBins)[0]
train_data[["FareBins","Survived"]].groupby(["FareBins"],as_index=False).mean().sort_values(by = "Survived",ascending = True)
fare = train_data




train_data.dropna(subset=["Embarked"],inplace=True,axis=0)
train_data["Embarked"].replace(["S","C","Q"],[0,1,2],inplace=True)
train_data["Sex"].replace(["female","male"],[0,1],inplace=True)
age_df=train_data[["Age",'Fare', 'Parch', 'SibSp', 'Pclass',"Embarked","Sex"]]
known_age = age_df[age_df.Age.notnull()].as_matrix()
unknown_age = age_df[age_df.Age.isnull()].as_matrix()
y=known_age[:,0]
x=known_age[:,1:]
rfr = RandomForestRegressor(random_state=0,n_estimators=2000,n_jobs=-1)
rfr.fit(x,y)
prediceAges = rfr.predict(unknown_age[:,1::])
score = rfr.score(x,y)#没有embarked 和sex  0.6462534980499668  有embarked sex 0.6964617145336285
train_data.loc[(train_data.Age.isnull()),"Age"]=prediceAges

