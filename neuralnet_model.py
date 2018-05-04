import os
import csv

import numpy as np
import pandas as pd

import xgboost
from sklearn.externals import joblib
from sklearn.utils import compute_class_weight
from sklearn.metrics import classification_report, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
# from keras import Sequential
# from keras import optimizers
# from keras.callbacks import ModelCheckpoint, EarlyStopping
# from keras.layers import Dropout, Dense
# from keras.models import load_model

def build_nn_model(x, y):
    model = Sequential()

    model.add(Dense(50, activation='sigmoid', input_shape=tuple(x.shape[1:])))
    model.add(Dropout(0.25))
    model.add(Dense(100, activation='sigmoid'))
    model.add(Dropout(0.25))
    # model.add(Dense(50, activation='relu'))
    # model.add(Dropout(0.25))
    model.add(Dense(y.shape[1], activation='softmax'))


    # op = optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)
    op = optimizers.Adam(lr=0.03)

    model.compile(optimizer=op, metrics=['categorical_accuracy'], loss='categorical_crossentropy')
    return model


def train_nn_model(x, y, model, model_name, weights):

    save_checkpoint = ModelCheckpoint(model_name, save_best_only=True, verbose=1)
    early_stop = EarlyStopping(min_delta=0.01, patience=200, verbose=1, mode='min')
    try:
        model.fit(x, y, batch_size=4000, epochs=100000, verbose=2, class_weight=weights,
                  callbacks=[save_checkpoint, early_stop], validation_split=0.2, shuffle=True)
    except Exception as e:
        print(e)


def build_logreg_model():
    model = LogisticRegression(penalty='l2', class_weight='balanced', random_state=1, solver='sag', max_iter=15,
                               multi_class='ovr', verbose=1, n_jobs=-1)
    return model


if __name__ == '__main__':
    os.chdir(r'C:\Users\lim_j\Google Drive\Technical Skills\GitHub\dmbi-datathon\data')

    # Cleaning out NANs and convert strings of numbers to numeric type
    # with open('hackathon_IoT_training_set_based_on_01mar2017.csv') as csv_file:
    #     data = list(csv.reader(csv_file))
    # df = pd.DataFrame(data[1:], columns=data[0])
    # ind = df.iloc[:, :-1]
    # dep = df.iloc[:, -1]
    # ind = ind.apply(lambda x: pd.to_numeric(x, errors='coerce'))
    # df2 = pd.concat([ind, dep], axis=1)
    # no_nans = df2.isnull().sum()
    # no_nans.to_csv('number of nans per column.csv')
    # cleaned_df = df2.dropna(axis=0, how='any')

    cleaned_df = joblib.load('cleaned_df.pkl')

    # x = cleaned_df.drop('device_category', axis=1)
    # y = cleaned_df['device_category']
    #
    # encoded_x = np.array(x)
    # label_bin = LabelBinarizer()
    # label_bin.fit(np.array(y))
    # encoded_y = label_bin.transform(y)
    # encoded_y = np.array(encoded_y)
    #
    # # encoded_df = np.concatenate([encoded_x, encoded_y], axis=1)
    #
    # x_train, x_test, y_train, y_test = train_test_split(encoded_x, y, test_size=0.2, random_state=1)
    #
    # # model = build_nn_model(encoded_x, encoded_y)
    #
    # weights = compute_class_weight('balanced', np.unique(y), y)
    #
    # # train_nn_model(encoded_x, encoded_y, model, 'dmbi-datathon-first-nn.h5', weights)
    #
    # model = build_logreg_model()
    #
    # model.fit(x_train, y_train)
    #
    # pred_y = model.predict_proba(x_test)
    #
    # report = classification_report(y_test, pred_y)
    #
    # logloss = log_loss(y_test, pred_y)

    all_y = cleaned_df['device_category']

    distinct_classes = np.unique(all_y).tolist()

    results = pd.DataFrame(columns=['class_name', 'max_prob'])

    results2 = []
    for left_out_class in distinct_classes:
        nine_classes = cleaned_df[cleaned_df['device_category'] != left_out_class]
        left_out_class_df = cleaned_df[cleaned_df['device_category'] == left_out_class]

        # Splitting dataset with one class out of dataset
        x_train = nine_classes.drop('device_category', axis=1)
        y_train = nine_classes['device_category']
        x_test = left_out_class_df.drop('device_category', axis=1)
        y_test = left_out_class_df['device_category']

        # RF MODEL
        # rf = RandomForestClassifier(100, criterion='gini', oob_score=True, n_jobs=-1,
        #                             random_state=1, class_weight='balanced')
        # rf.fit(x_train, y_train)
        # y_pred = rf.predict_proba(x_test)

        # LOG REG MODEL
        model = build_logreg_model()
        model.fit(x_train, y_train)

        # y_pred = model.predict(x_test)
        #
        # report = classification_report(y_test, y_pred)
        #
        # logloss = log_loss(y_test, y_pred)

        probs = model.predict_proba(x_test)

        highest_prob = float(max([max(sample) for sample in probs]))

        # results = results.append(pd.DataFrame([left_out_class, highest_prob]), ignore_index=True)

        results2.append({'left_out_class': left_out_class, 'highest_prob': highest_prob})

    # results.to_csv('oob_class_logreg_highest_probs.csv')

    pd.DataFrame(results2).to_csv('oob_class_logreg_high_probs.csv')

    print('test')