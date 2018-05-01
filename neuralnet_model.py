from keras import Sequential
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dropout, Dense


def build_model(x, y):
    model = Sequential()

    model.add(Dense(150, activation='relu', input_shape=tuple(x.shape[1:])))
    model.add(Dropout(0.25))
    model.add(Dense(300, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(300, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(y.shape[1], activation='softmax'))

    op = optimizers.Adam()
    model.compile(optimizer=op, metrics=['categorical_accuracy', 'sparse_categorical_accuracy'],
                  loss='categorical_crossentropy')
    return model


def train_model(x, y, model, model_name):
    save_checkpoint = ModelCheckpoint(model_name, save_best_only=True, verbose=1)
    early_stop = EarlyStopping(min_delta=0.01, patience=200, verbose=1, mode='min')
    try:
        model.fit(x, y, batch_size=2000, epochs=100000, verbose=2,
                  callbacks=[save_checkpoint, early_stop], validation_split=0.2, shuffle=True)
    except Exception as e:
        print(e)


if __name__ == '__main__':
    print('template model')
