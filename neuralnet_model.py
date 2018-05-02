from sklearn.utils import compute_class_weight
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer
from sklearn.linear_model.logistic import LogisticRegression
import numpy as np
from keras import Sequential
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dropout, Dense


def build_nn_model(x, y):
    model = Sequential()

    model.add(Dense(25, activation='relu', input_shape=tuple(x.shape[1:])))
    model.add(Dropout(0.25))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(y.shape[1], activation='softmax'))

    op = optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)
    model.compile(optimizer=op, metrics=['categorical_accuracy', 'sparse_categorical_accuracy'],
                  loss='categorical_crossentropy')
    return model


def train_nn_model(x, y, model, model_name):

    class_weight = compute_class_weight('balanced', np.unique(y), y)
    save_checkpoint = ModelCheckpoint(model_name, save_best_only=True, verbose=1)
    early_stop = EarlyStopping(min_delta=0.01, patience=200, verbose=1, mode='min')
    try:
        model.fit(x, y, batch_size=5000, epochs=100000, verbose=2, class_weight=class_weight,
                  callbacks=[save_checkpoint, early_stop], validation_split=0.2, shuffle=True)
    except Exception as e:
        print(e)


def build_logreg_model(x, y):
    model = LogisticRegression(penalty='l2', class_weight='balanced', random_state=1, solver='sag', max_iter=100,
                               multi_class='multinomial', verbose=1, n_jobs=-1)
    return model

if __name__ == '__main__':
    print('template model')
