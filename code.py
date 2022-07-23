import pandas as pd
columnames=["Id","Class","Sex","Age","SibSp","Parch","Survived"]
df=pd.read_csv("titanic.csv",names=columnames).iloc[1:]
print(df.head())

features=["Id","Class","Sex","Age","SibSp","Parch","Survived"]
X=df[features]
y=df.Survived
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1)
clf=DecisionTreeClassifier()
clf=clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test,y_pred)*100)

from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus
dot_data=StringIO()
export_graphviz(clf,out_file=dot_data,filled=True,rounded=True,special_characters=True,feature_names=features,class_names=['0','1'])
print(dot_data.getvalue())

graph=pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png("Survival.png")
Image(graph.create_png())

clf=DecisionTreeClassifier(max_depth=3)
clf=clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test,y_pred)*100)

dot_data=StringIO()
export_graphviz(clf,out_file=dot_data,filled=True,rounded=True,special_characters=True,feature_names=features,class_names=['0','1'])
graph=pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png("Survival.png")
Image(graph.create_png())