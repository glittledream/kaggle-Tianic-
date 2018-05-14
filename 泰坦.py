import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

train_data = pd.read_csv("/home/gch/桌面/train.csv")
#探索数据
fig=plt.figure(figsize=(10,10))
fig.set(alpha=0.2)
plt.subplot2grid((2,3),(0,0))
data.Survived.value_counts().plot(kind="bar")
plt.title(u"Survived(1 rescued)")
plt.ylabel("people counting")

plt.subplot2grid((2,3),(0,1))
data.Pclass.value_counts().plot(kind="bar")
plt.title(u"pcalss")
plt.ylabel(u"passenger")

plt.subplot2grid((2,3),(0,2))
plt.scatter(data.Survived,data.Age)
plt.grid(b=True, which='major', axis='y')
plt.ylabel("ages")
plt.title("age rescued 1")

plt.subplot2grid((2,3),(1,0),colspan=2)
data.Age[data.Pclass ==1].plot(kind="kde")
data.Age[data.Pclass ==2].plot(kind="kde")
data.Age[data.Pclass ==3].plot(kind="kde")
plt.legend(labels=["1","2","3"])#1/2/3等仓位
plt.xlabel("age")
plt.ylabel("density")
plt.title("Pclass age distrubte")

plt.subplot2grid((2,3),(1,2))
data.Embarked.value_counts().plot(kind='bar')
plt.ylabel("people counting")
plt.title("people landing on board")
plt.show()

#各乘客等级获救情况
fig=plt.figure()
fig.set(alpha=0.2)
survived_0 = data.Pclass[data.Survived == 0].value_counts()
survived_1 = data.Pclass[data.Survived == 1].value_counts()
df =pd.DataFrame({"unrescued":survived_0,"rescued":survived_1})
df.plot(kind="bar",stacked=True)
plt.title("condition")
plt.ylabel("number of people")
plt.xlabel("rank")
plt.show()

#各乘客性格的获救情况
fig =plt.figure()
fig.set(alpha=0.2)
Surivived_m = data.Survived[data.Sex == "male"].value_counts()
Surivived_f = data.Survived[data.Sex == 'female'].value_counts()
df1 = pd.DataFrame({"male":Surivived_m,"female":Surivived_f})
df1.plot(kind="bar",stacked=True)
plt.xlabel("SEX")
plt.ylabel("number of people")
plt.title("sex condition")


# 各种舱级别情况下各性别的获救情况
fig=plt.figure()
ax1=fig.add_subplot(141)
data.Survived[data.Sex == 'female'][data.Pclass == 1].value_counts().plot(kind="bar",label="female high class")
ax1.legend(loc="best")

ax3=fig.add_subplot(142,sharey=ax1)
data.Survived[data.Sex == "female"][data.Pclass != 1].value_counts().plot(kind="bar",label="female low class")
ax3.legend(loc="best")

ax2=fig.add_subplot(143,sharey=ax1)
data.Survived[data.Sex == "male"][data.Pclass == 1].value_counts().plot(kind="bar",label="male high class")
ax2.legend(loc="best")

ax4=fig.add_subplot(144,sharey=ax1)
data.Survived[data.Sex == "male"][data.Pclass != 1].value_counts().plot(kind="bar",label="male low class")
ax4.legend(loc="best")

#进口点
fig = plt.figure()
fig.set(alpha=0.2)
Survived_0 = data.Embarked[data.Survived == 0].value_counts()
Survived_1 = data.Embarked[data.Survived == 1].value_counts()
df=pd.DataFrame({'rescued':Survived_1, 'unrescued':Survived_0})
df.plot(kind='bar', stacked=True)
plt.title(u"Emakred condition")
plt.xlabel(u"Emakred")
plt.ylabel(u"number of people")
plt.show()

#堂兄弟/妹，孩子/父母有几人，对是否获救的影响。
g = data.groupby(["SibSp","Survived"])
df = pd.DataFrame(g.count()["PassengerId"])
df1 = pd.DataFrame(g.count()["Cabin"])
print(df,df1)

#简单的预处理
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# 把已有的数值型特征取出来丢进Random Forest Regressor中
train_data = train_data
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

#####训练数据集
import seaborn as sb
from sklearn.model_selection import train_test_split,cross_validate,cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV,StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
df2 = train_data
all_classes = df2["Survived"]
all_inputs = df2[["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]]
(x_train,
 x_test,
 y_train_class,
 y_test_class) = train_test_split(all_inputs,all_classes,random_state=1,train_size=0.75)
decision_treeclassifier = DecisionTreeClassifier()
decision_treeclassifier.fit(x_train,y_train_class)
decision_treeclassifier.score(x_test,y_test_class)
cv_scores = cross_validate(decision_treeclassifier,all_inputs,all_classes,cv=10)
print(cv_scores)#可以看出训练时间和分数
cv_scores1 = cross_val_score(decision_treeclassifier,all_inputs,all_classes,cv=10)
print(cv_scores1)#十折分数有高有低，说明有调参空间
sb.distplot(cv_scores1)
plt.title('Average score: {}'.format(np.mean(cv_scores1)))

#搜索最好的模型   先要得到descision_classfier = DescisionClassfier
param_search = {"max_depth":[1,2,3,4,5],
                "max_features":[1,2,3,4]}
cross_v = StratifiedKFold(n_splits=10)
grid_search = GridSearchCV(decision_treeclassifier,param_grid=param_search,cv=cross_v)
grid_search.fit(all_inputs,all_classes)
print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))
#Best score: 0.8076490438695163
#Best parameters: {'max_depth': 4, 'max_features': 3}

#使用随机森林
random_forest = RandomForestClassifier()
param_search = {"n_estimators":[5,10,25,50,75],
                "max_features":[1,2,3,4,5],
                "max_depth":[1,2,3.4,5],
                "criterion":["gini","entropy"],
                "warm_start":[True,False]}
cross_v = StratifiedKFold(n_splits=10)
grid_search = GridSearchCV(random_forest,param_grid=param_search,cv=cross_v)
grid_search.fit(all_inputs,all_classes)
print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))

#测试数据
test_data = pd.read_csv("/home/gch/桌面/test.csv")
test_data.info()
test_data["Fare"].fillna(method="bfill",inplace=True)
test_data["Embarked"].replace(["S","C","Q"],[0,1,2],inplace=True)
test_data["Sex"].replace(["female","male"],[0,1],inplace=True)
age_df=test_data[["Age",'Fare', 'Parch', 'SibSp', 'Pclass',"Embarked","Sex"]]
known_age = age_df[age_df.Age.notnull()].as_matrix()
unknown_age = age_df[age_df.Age.isnull()].as_matrix()
y=known_age[:,0]
x=known_age[:,1:]
rfr = RandomForestRegressor(random_state=0,n_estimators=2000,n_jobs=-1)
rfr.fit(x,y)
prediceAges = rfr.predict(unknown_age[:,1::])
test_data.loc[(test_data.Age.isnull()),"Age"]=prediceAges


#预测
ALL_inputs = test_data[["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]].values
random_forest = RandomForestClassifier(criterion='gini', max_features= 4, n_estimators= 10,max_depth=5, warm_start=True)
random_forest.fit(all_inputs,all_classes)
predict = random_forest.predict(ALL_inputs)
result = pd.DataFrame({"PassengerId":test_data["PassengerId"],"Survived":predict},columns=["PassengerId","Survived"])
result.to_csv("/home/gch/桌面/randomforst.csv")




