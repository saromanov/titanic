import argparse
import numpy as np
import pandas as pd
from sklearn import preprocessing, cross_validation
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn import svm
from scipy.stats import mode
from sklearn.grid_search import GridSearchCV
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import np_utils

def load_test():
    df = pd.read_csv('test.csv')
    df = df.drop(['Name', 'Ticket', 'Cabin'], axis=1)
    df['Gender']= df['Sex'].map({'female':0, 'male': 1}).astype(int)
    age_mean = df['Age'].mean()
    df['Age'] = df['Age'].fillna(age_mean)
    fare_means = df.pivot_table('Fare', index='Pclass', aggfunc='mean')


def nn(X, y, Xtest, ytest):
    print("THis is shape: ", X.shape, y.shape, np.max(y)+1)
    y = np_utils.to_categorical(y, np.max(y)+1)
    ytest = np_utils.to_categorical(ytest, np.max(y)+1)
    model = Sequential()
    model.add(Dense(9,10))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.2))
    model.add(Dense(10,6))
    model.add(Activation('relu'))
    model.add(Dense(6,np.max(y)+1))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    history = model.fit(X, y, nb_epoch=20, verbose=1, show_accuracy=True, validation_split=0.1)
    score = model.evaluate(Xtest, ytest, verbose=1, show_accuracy=True)
    print('Test score: ', score[0])

def rf(X,y, xtest, ytest):
    anova_pre = SelectKBest(f_regression, k=8)
    model = Pipeline([
        ('inp', preprocessing.Imputer(strategy='mean', missing_values=-1)),
        ('anova', anova_pre),
        ('clf', GradientBoostingClassifier()),
        ])
    grid = GridSearchCV(model, {
        'inp__strategy': ['mean', 'median'],
        'clf__learning_rate': [0.5, 0.8,1],
        'clf__max_depth':[5, 7, None],
        'clf__n_estimators': [50, 100],
        }, cv=5, verbose=3)
    model = grid.fit(X, y)
    output = model.predict(test_data[:,1:])
    print("This is test: ", model.score(xtest, ytest))


def model1(title):
    df = pd.read_csv('./data/train.csv')
    df = df.drop(['Name', 'Ticket', 'Cabin'], axis=1)
    df['Gender']= df['Sex'].map({'female':0, 'male': 1}).astype(int)
    age_mean = df['Age'].mean()
    mode_embarked = mode(df['Embarked'])[0][0]
    df['Embarked'] = df['Embarked'].fillna(mode_embarked)
    df['Age'] = df['Age'].fillna(age_mean)
    df = pd.concat([df, pd.get_dummies(df['Embarked'], prefix='Embarked')], axis=1)
    df = df.drop(['Sex', 'Embarked'], axis=1)
    cols = df.columns.tolist()
    cols = [cols[1]] + cols[0:1] + cols[2:]
    df = df[cols]
    train_data = df.values
    #rf(train_data[0:, 2:], train_data[0:,0], train_data[0:, 2:], train_data[0:,0])
    #model = RandomForestClassifier(n_estimators=100)
    #model = model.fit(train_data[0:, 2:], train_data[0:,0])

    df_test = pd.read_csv('./data/test.csv')
    df_test = df_test.drop(['Name', 'Ticket', 'Cabin'], axis=1)
    df_test['Gender']= df_test['Sex'].map({'female':0, 'male': 1}).astype(int)
    age_mean = df_test['Age'].mean()
    df_test['Age'] = df_test['Age'].fillna(age_mean)
    fare_means = df.pivot_table('Fare', index='Pclass', aggfunc='mean')
    df_test['Fare'] = df_test[['Fare', 'Pclass']].apply(lambda x: fare_means[x['Pclass']]
        if pd.isnull(x['Fare']) else x['Fare'], axis=1)
    df_test = pd.concat([df_test, pd.get_dummies(df_test['Embarked'], prefix='Embarked')], axis=1)
    df_test = df_test.drop(['Sex', 'Embarked'], axis=1)
    test_data = df_test.values
    if title.rf:
        rf(train_data[0:, 2:], train_data[0:,0], train_data[0:, 2:], train_data[0:,0])
    if title.mlp:
        nn(train_data[0:, 2:], train_data[0:,0], train_data[0:, 2:], train_data[0:,0])


def run():
    parser = argparse.ArgumentParser(description='Parsing model arguments')
    parser.add_argument('--rf', help='Random forest model', default=False, action='store_true')
    parser.add_argument('--mlp', help='MLP model', default=False, action='store_true')
    result = parser.parse_args()
    model1(result)


run()

