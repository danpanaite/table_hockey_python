# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
def get_query(cursor, first, season, strength):
    cursor_filter = (
        '(after: "{}", first: {})'.format(cursor, first) if cursor is not None else ""
    )
    strength_filter = "strength: {}".format(strength) if strength is not None else ""
    shots_filter = "(filter: {%s})" % strength_filter if strength is not None else ""
    season_filter = '(filter: {id: "%s" })' % season

    return """
    {
      seasons%s {
        shots%s {
          fenwickEvents%s {
            total
            pageInfo {
              startCursor
              endCursor
              hasNextPage
            }
            edges {
              node {
                type
                period
                homeGame
                coordX
                coordY
                angle
                distance
                shotType
                event
                priorEvent
                team
                priorEventTeam
                angleChange
                distanceChange
                leading
                leadingByOne
                trailing
                trailingByOne
                secondsChange
              }
            }
          }
        }
      }
    }
    """ % (
        season_filter,
        shots_filter,
        cursor_filter,
    )


print(get_query(cursor=None, first=10000, season="20102011", strength="EVENSTRENGTH"))

# %%
def get_shots(url, cursor, first, season, strength):
    response = requests.post(
        url,
        json={
            "query": get_query(
                cursor=cursor, first=first, season=season, strength=strength
            )
        },
    )
    json_data = json.loads(response.text)
    total = json_data["data"]["seasons"][0]["shots"]["fenwickEvents"]["total"]
    has_next_page = json_data["data"]["seasons"][0]["shots"]["fenwickEvents"]["pageInfo"]["hasNextPage"]
    cursor = json_data["data"]["seasons"][0]["shots"]["fenwickEvents"]["pageInfo"]["endCursor"]
    df_data = json_data["data"]["seasons"][0]["shots"]["fenwickEvents"]["edges"]
    df_data = list(map(lambda edge: edge["node"], df_data))
    df = pd.DataFrame(df_data)

    return df, total, has_next_page, cursor


# %%
import requests
import json
import pandas as pd
from tqdm import tqdm

first = 5000
strength = "EVENSTRENGTH"
url = "http://localhost:4000/"

seasons = [
    "20102011",
    "20112012",
    "20122013",
    "20132014",
    "20142015",
    "20152016",
    "20162017",
    "20172018",
    "20182019",
    "20192020",
]
df = pd.DataFrame()

for season in seasons:
    print(f"Grabbing data for {season}")
    df_season, total, has_next_page, cursor = get_shots(
        url, None, first, season, strength
    )

    pbar = tqdm(total=total)

    while has_next_page is True:
        df_new, total, has_next_page, cursor = get_shots(
            url, cursor, first, season, strength
        )
        df_season = df_season.append(df_new)

        pbar.update(len(df_new))

    pbar.close()
    df = df.append(df_season)


# %%
import numpy as np

df["priorEventShot"] = df["priorEvent"] == "SHOT"
df["priorEventSameTeam"] = df["priorEventTeam"] == df["team"]
df["snapShot"] = df["shotType"] == "SNAP SHOT"
df["wristShot"] = df["shotType"] == "WRIST SHOT"
df["slapShot"] = df["shotType"] == "SLAP SHOT"
df["deflected"] = df["shotType"] == "DEFLECTED"
df["backhand"] = df["shotType"] == "BACKHAND"
df["wrapAround"] = df["shotType"] == "WRAP-AROUND"
df["tipIn"] = df["shotType"] == "TIP-IN"
df["goal"] = df["type"] == "GOAL"
df = df.dropna()

feature_names = [
    "priorEventShot",
    "priorEventSameTeam",
    "coordX",
    "coordY",
    "angle",
    "distance",
    "snapShot",
    "wristShot",
    "slapShot",
    "deflected",
    "backhand",
    "wrapAround",
    "tipIn",
    "angleChange",
    "distanceChange",
    "leading",
    "leadingByOne",
    "trailing",
    "trailingByOne",
    "secondsChange",
]

inputs = df[feature_names].dropna().to_numpy().astype("float32")
outputs = df[["goal"]].to_numpy().flatten().astype("float32")

print(feature_names)
print(np.isnan(inputs).any())
print(np.isnan(outputs).any())
print(len(inputs))


# %%
def normalize(inputs):
    inputs_min = np.min(inputs, axis=0)
    inputs_max = np.max(inputs, axis=0)

    return (inputs - inputs_min) / (inputs_max - inputs_min)


# %%
# inputs = normalize(inputs)

features_length = len(np.array(inputs)[0, :])
inputs_train_length = np.ceil(len(inputs[:, 0]) * 0.9).astype("int")
inputs_train = inputs[0:inputs_train_length]
inputs_test = inputs[inputs_train_length:]

outputs_train = outputs[0:inputs_train_length]
outputs_test = outputs[inputs_train_length:]

# print(len(outputs_test))
# print(inputs)
# print(np.mean(inputs, axis=0))
# print(np.var(inputs, axis=0))


# %%
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

model_file = "test.h5"

layers = [features_length, features_length * 5, 1]
epochs = 200
batch_size = 100


def build_model():
    model = Sequential()

    model.add(
        Dense(
            layers[1],
            input_dim=layers[0],
            activation=LeakyReLU(alpha=0.3),
            # kernel_regularizer = regularizers.l2(0.001)
        )
    )

    # model.add(Dropout(0.2))

    # model.add(Dense(
    #     layers[1],
    #     activation="relu"
    # ))

    # model.add(Dropout(0.2))

    model.add(Dense(layers[2], activation="sigmoid"))

    start = time.time()

    # optimizer = SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    print("> Compilation Time : ", time.time() - start)
    return model


def train_model(model_file, epochs, layers, batch_size):
    global_start_time = time.time()

    # tensorboard = TensorBoard(log_dir="logs/"+ str(global_start_time),
    #     histogram_freq=5,
    #     write_graph= True,
    #     write_grads = True)

    model = KerasRegressor(
        build_model, epochs=epochs, batch_size=batch_size, validation_split=0.1
    )
    history = model.fit(inputs_train, outputs_train)

    # model = build_model()
    # history= model.fit(inputs_train, outputs_train, epochs=epochs, batch_size=batch_size, validation_split=0.1)

    # kfold = StratifiedKFold(n_splits=10, shuffle=True)
    # results = cross_val_score(model, inputs_train, outputs_train, cv=kfold)

    # model.save(model_file)

    print("Training duration (s) : ", time.time() - global_start_time)

    return [model, history]


[model, history] = train_model(model_file, epochs, layers, batch_size)

# layer = model.get_layer(index=0)
# weights = layer.get_weights()


# %%
predictions = model.predict(inputs_test)

print(predictions)
print(np.max(predictions))
print(np.sum(outputs_test))


# %%
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt

predictions = model.predict(inputs_test)

fpr_keras, tpr_keras, thresholds_keras = roc_curve(outputs_test, predictions)
auc_keras = auc(fpr_keras, tpr_keras)

plt.figure(1)
plt.plot([0, 1], [0, 1], "k--")
plt.plot(fpr_keras, tpr_keras, label="Keras (area = {:.3f})".format(auc_keras))
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("ROC curve")
plt.legend(loc="best")
plt.show()


# %%
import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(model, random_state=1).fit(inputs_test, outputs_test)
eli5.show_weights(perm, feature_names=feature_names)


# %%
xl


# %%
