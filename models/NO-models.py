import numpy as np
from sklearn.metrics import confusion_matrix, recall_score, precision_score, plot_confusion_matrix, multilabel_confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report
import lightgbm
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier
from keras import regularizers
import warnings
warnings.filterwarnings('ignore')


# function to evaluate predictions
def evaluate(y_true, y_pred,multi=False):
    # calculate and display confusion matrix
    labels = np.unique(y_true)
    if multi==False:
        cm = confusion_matrix(y_true, y_pred, labels=labels)
    else:
        cm = multilabel_confusion_matrix(y_true, y_pred)
    # calculate precision, recall, and F1 score
    accuracy = float(np.trace(cm)) / np.sum(cm)
    precision = precision_score(y_true, y_pred, average=None, labels=labels)[1]
    recall = recall_score(y_true, y_pred, average=None, labels=labels)[1]
    f1 = 2 * precision * recall / (precision + recall)
    print("accuracy:", accuracy)
    print("precision:", precision)
    print("recall:", recall)
    print("f1 score:", f1)




def binary_models(model,x_train,y_train,x_test,y_test,x_val=False,y_val=False,gen=False):
    if model=='lr':
        LogReg=LogisticRegression(max_iter=100000)
        LogReg.fit(x_train, y_train)
        model_=LogReg
    elif model=='svc':
        sv=SVC()
        sv.fit(x_train,y_train)
        model_=sv
    elif model=='dtc':
        dt=DecisionTreeClassifier()
        dt.fit(x_train, y_train)
        model_=dt
    elif model=='rft':
        rf=RandomForestClassifier()
        rf.fit(x_train, y_train)
        model_=rf
    elif model=='lgbc':
        lgbc=lightgbm.LGBMClassifier(n_estimators=500, learning_rate=0.1, num_leaves=32, colsample_bytree=0.2, reg_alpha=3, reg_lambda=1, min_split_gain=0.01, min_child_weight=40, num_threads=2)
        lgbc.fit(x_train, y_train)
        model_=lgbc
    elif model=='gbc':
        gbc=GradientBoostingClassifier(n_estimators=500, learning_rate=0.1, max_depth=2)
        gbc.fit(x_train, y_train)
        model_=gbc
    elif model=='xgbc':
        xgbc=XGBClassifier(n_estimators=500, learning_rate=0.1, max_depth=2)
        xgbc.fit(x_train, y_train)
        model_=xgbc
    elif model=='nn':
        model = Sequential()
        model.add(Dense(150, input_dim=len(x_train.columns), activation='relu'))
        model.add(Dropout(rate=0.5))
        model.add(Dense(100, activation='relu'))
        model.add(Dropout(rate=0.5))
        model.add(Dense(50, activation='relu'))
        model.add(Dropout(rate=0.5))
        model.add(Dense(1, activation='sigmoid'))
        print(model.summary())
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=5, batch_size=128, validation_data=(x_val,y_val))
        # evaluate the keras model
        y_pred = model.predict(x_val)
        y_pred = (y_pred > 0.5).astype(int)
        model.nn_results(model,x_train,y_train)
        model.nn_results(model,x_val,y_val)
        model.nn_results(model,x_test,y_test)
        model_=model
        return model_
    elif model=='nnReg':
        model = Sequential()
        model.add(Dense(150, input_dim=len(x_train.columns), activation='relu', kernel_regularizer=regularizers.l2(0.001)))
        model.add(Dropout(rate=0.5))
        model.add(Dense(100, activation='relu'))
        model.add(Dropout(rate=0.5))
        model.add(Dense(50, activation='relu'))
        model.add(Dropout(rate=0.5))
        model.add(Dense(1, activation='sigmoid'))
        print(model.summary())
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=5, batch_size=128, validation_data=(x_val,y_val))
        # evaluate the keras model
        y_pred = model.predict(x_val)
        y_pred = (y_pred > 0.5).astype(int)
        print('Train results')
        model.nn_results(model,x_train,y_train)
        print('Vaildation results')
        model.nn_results(model,x_val,y_val)
        print('Test results')
        model.nn_results(model,x_test,y_test)
        model_=model
        return model_
    ypred=model_.predict(x_test)
    if gen==True:
        print(classification_report(y_test, ypred))
        return model_

    print('train score',model_.score(x_train, y_train))
    evaluate(y_test, ypred)
    plot_confusion_matrix(model_, x_test, y_test)
    print(classification_report(y_test, ypred))
    print('Confusion matrix\n- x-axis is true labels (none, comp1, etc.)\n- y-axis is predicted labels')  
    return model_