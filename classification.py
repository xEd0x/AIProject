
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.utils.fixes import loguniform

import keras
from keras.models import Sequential
from keras.layers import Dense

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

dataset = pd.read_csv('train.csv')

#Pearson correlation
plt.figure(figsize = (20,20))
cor = dataset.corr()
sns.heatmap(cor, annot = True, cmap = plt.cm.Reds)
#plt.savefig('pearson.png')
plt.show()

dataset.drop('three_g', axis = 1, inplace = True)
dataset.drop('pc', axis = 1, inplace = True)

print(dataset.head(10))

X = dataset.iloc[:,:18].values
y = dataset.iloc[:,18:19].values

sc = StandardScaler()
X = sc.fit_transform(X)


ohe = OneHotEncoder()
y = ohe.fit_transform(y).toarray()


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.1)


# Neural network
model = Sequential()
model.add(Dense(16, input_dim=18, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(4, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train,validation_data = (X_test,y_test), epochs=100, batch_size=64)

y_pred = model.predict(X_test)

pred = list()
for i in range(len(y_pred)):
    pred.append(np.argmax(y_pred[i]))

test = list()
for i in range(len(y_test)):
    test.append(np.argmax(y_test[i]))


a = accuracy_score(pred,test)
print('Neural Network Accuracy:', a*100)


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(history.history['loss']) 
plt.plot(history.history['val_loss']) 
plt.title('Model loss') 
plt.ylabel('Loss') 
plt.xlabel('Epoch') 
plt.legend(['Train', 'Test'], loc='upper left') 
plt.show()


#Logistic Regression
modelL = LogisticRegression()

label_encoder = preprocessing.LabelEncoder()
dataset = dataset.apply(label_encoder.fit_transform)

logistic_y = dataset['price_range']
logistic_x = dataset

logistic_x.drop('price_range', axis = 1, inplace = True)

sc = StandardScaler()
logistic_x = sc.fit_transform(logistic_x)

logistic_x_train, logistic_x_test, logistic_y_train, logistic_y_test = train_test_split(logistic_x,logistic_y, random_state=0)
modelL.fit(logistic_x_train, logistic_y_train)

scoreL = modelL.score(logistic_x_test, logistic_y_test)

logistic_y_pred = modelL.predict(logistic_x_test)

print("Logistic Regression Accuracy: ", scoreL*100)

cnf_matrix = metrics.confusion_matrix(logistic_y_test, logistic_y_pred)

class_names=[0,1,2,3]
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

#DecisionTree
dtree_model = DecisionTreeClassifier(max_depth = 10).fit(X_train, y_train) 
dtree_predictions = dtree_model.predict(X_test)

dtree_accuracy = accuracy_score(y_test, dtree_predictions)

print("Decision Tree Accuracy: ", dtree_accuracy * 100)


#SVM
svm_model_linear = OneVsRestClassifier(SVC(kernel = 'linear', C = 20, probability = True)).fit(X_train, y_train)
svm_predictions_linear = svm_model_linear.predict(X_test)

svm_model_rbf = OneVsRestClassifier(SVC(kernel = 'rbf', C = 20)).fit(X_train, y_train)
svm_predictions_rbf = svm_model_rbf.predict(X_test)

svm_model_poly = OneVsRestClassifier(SVC(kernel = 'poly', C = 10, gamma = 'scale', coef0 = 2, degree = 2, break_ties=True, tol = 0.1)).fit(X_train, y_train)
svm_predictions_poly = svm_model_poly.predict(X_test)

svm_model_sigmoid = OneVsRestClassifier(SVC(kernel = 'sigmoid', C = 5)).fit(X_train, y_train)
svm_predictions_sigmoid = svm_model_sigmoid.predict(X_test)

svm_accuracy_linear = svm_model_linear.score(X_test, y_test)
svm_accuracy_rbf = svm_model_rbf.score(X_test, y_test)
svm_accuracy_poly = svm_model_poly.score(X_test, y_test)
svm_accuracy_sigmoid = svm_model_sigmoid.score(X_test, y_test)

print ("SVM Accuracy (linear kernel): ", svm_accuracy_linear * 100)
print ("SVM Accuracy (rbf kernel): ", svm_accuracy_rbf * 100)
print ("SVM Accuracy (poly kernel): ", svm_accuracy_poly * 100)
print ("SVM Accuracy (sigmoid kernel): ", svm_accuracy_sigmoid * 100)