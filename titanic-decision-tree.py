
import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

from sklearn.ensemble import BaggingClassifier


data = pd.read_csv('titanic.csv')



columns_target = ['Survived'] 

columns_train = ['Pclass', 'Sex', 'Age', 'Fare']



X = data[columns_train]
Y = data[columns_target]



# Проверяем есть ли пустые ячейки в колонках




X['Sex'].isnull().sum()




X['Pclass'].isnull().sum()




X['Fare'].isnull().sum()




X['Age'].isnull().sum()




# Заполняем пустые ячейки медианным значением по возрасту




pd.options.mode.chained_assignment = None 



X['Age'] = X['Age'].fillna(X['Age'].mean())




X['Age'].isnull().sum()




# Заменяем male и female на 0 и 1 с помощью словаря




d={'male':0, 'female':1} # создаем словарь




X['Sex'] = X['Sex'].apply(lambda x:d[x])




X['Sex'].head() 



#Разделяем нашу выборку на обучающую и тестовую



from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)




clf = tree.DecisionTreeClassifier (max_depth=5, random_state=21) 




bagging = BaggingClassifier(base_estimator=clf,n_estimators=100)



bagging.fit(X_train,Y_train)



bagging.score (X_test,Y_test)



clf.fit(X_train, Y_train) # обучаем модель



clf.score(X_test,Y_test) # проверяем точность



import eli5



eli5.explain_weights_sklearn(clf, feature_names=X_train.columns.values)




# Загружаем модель Support VEctor Machine для обучения



from sklearn import svm



predmodel = svm.LinearSVC()



# Обучаем модель с помощью нашей обучающей выборки



predmodel.fit(X_train, Y_train)



# Предсказываем на тестовой выборке



predmodel.predict(X_test[0:10])



# Проверяем точность предсказаний



predmodel.score(X_test,Y_test)
