import numpy as np
import time
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from keras.layers.advanced_activations import LeakyReLU

from keras.losses import BinaryCrossentropy
from keras.layers.core import Dense, Activation, Dropout
from keras.models import Sequential
from keras import regularizers
from keras.callbacks import TensorBoard
from keras.optimizers import SGD


def xg_sequential(shots_data, feature_names, epochs, batch_size):
    features_length, inputs_train, inputs_test, outputs_train, outputs_test = normalize_shots_data(
        shots_data, feature_names)

    layers = [features_length, features_length - 5, 1]

    [model, _] = train_model(epochs, layers, batch_size, inputs_train, outputs_train)

    return model, features_length, inputs_train, inputs_test, outputs_train, outputs_test


def normalize_shots_data(df, feature_names):
    df['priorEventShot'] = df['priorEvent'] == "SHOT"
    df['priorEventSameTeam'] = df['priorEventTeam'] == df['team']
    df['snapShot'] = df['shotType'] == "SNAP SHOT"
    df['wristShot'] = df['shotType'] == "WRIST SHOT"
    df['slapShot'] = df['shotType'] == "SLAP SHOT"
    df['deflected'] = df['shotType'] == "DEFLECTED"
    df['backhand'] = df['shotType'] == "BACKHAND"
    df['wrapAround'] = df['shotType'] == "WRAP-AROUND"
    df['tipIn'] = df['shotType'] == "TIP-IN"
    df['goal'] = df['type'] == 'GOAL'
    df = df.dropna()

    inputs = df[feature_names].dropna().to_numpy().astype('float32')
    outputs = df[['goal']].to_numpy().flatten().astype('float32')

    features_length = len(np.array(inputs)[0, :])
    inputs_train_length = np.ceil(len(inputs[:, 0]) * 0.9).astype('int')
    inputs_train = inputs[0: inputs_train_length]
    inputs_test = inputs[inputs_train_length:]

    outputs_train = outputs[0: inputs_train_length]
    outputs_test = outputs[inputs_train_length:]

    return features_length, inputs_train, inputs_test, outputs_train, outputs_test


def build_model(layers):
    model = Sequential()

    model.add(Dense(
        layers[1],
        input_dim=layers[0],
        activation=LeakyReLU(alpha=0.3),
        # kernel_regularizer = regularizers.l2(0.001)
    ))

    # model.add(Dropout(0.2))

    # model.add(Dense(
    #     layers[1],
    #     activation="relu"
    # ))

    # model.add(Dropout(0.2))

    model.add(Dense(
        layers[2],
        activation="sigmoid"
    ))

    start = time.time()

    # optimizer = SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
    model.compile(loss='binary_crossentropy',
                  optimizer="adam", metrics=["accuracy"])
    print("> Compilation Time : ", time.time() - start)
    return model


def train_model(epochs, layers, batch_size, inputs_train, outputs_train, model_file=None):
    global_start_time = time.time()

    # tensorboard = TensorBoard(log_dir="logs/"+ str(global_start_time),
    #     histogram_freq=5,
    #     write_graph= True,
    #     write_grads = True)

    model = KerasRegressor(lambda: build_model(
        layers), epochs=epochs, batch_size=batch_size, validation_split=0.1)
    history = model.fit(inputs_train, outputs_train)

    #model = build_model()
    #history= model.fit(inputs_train, outputs_train, epochs=epochs, batch_size=batch_size, validation_split=0.1)

    # kfold = StratifiedKFold(n_splits=10, shuffle=True)
    # results = cross_val_score(model, inputs_train, outputs_train, cv=kfold)

    # model.save(model_file)

    print('Training duration (s) : ', time.time() - global_start_time)

    return [model, history]