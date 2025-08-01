import os  
import numpy as np 
import cv2 
from tensorflow.keras.utils import to_categorical
from keras.layers import Input, Dense 
from keras.models import Model

is_init = False
size = -1

label = []
dictionary = {}
c = 0

for i in os.listdir():
    if i.split(".")[-1] == "npy" and not(i.split(".")[0] == "labels"):  
        data = np.load(i)
        print(f"Loaded {i} with shape: {data.shape}")  # Debugging line
        if data.size == 0:
            print(f"Warning: {i} is empty.")
            continue  # Skip empty arrays

        if not(is_init):
            is_init = True 
            X = data
            size = X.shape[0]
            y = np.array([i.split('.')[0]] * size).reshape(-1, 1)
        else:
            X = np.concatenate((X, data))
            y = np.concatenate((y, np.array([i.split('.')[0]] * size).reshape(-1, 1)))

        label.append(i.split('.')[0])
        dictionary[i.split('.')[0]] = c  
        c += 1

if is_init:
    for i in range(y.shape[0]):
        y[i, 0] = dictionary[y[i, 0]]
    y = np.array(y, dtype="int32")

    y = to_categorical(y)

    X_new = X.copy()
    y_new = y.copy()
    counter = 0 

    cnt = np.arange(X.shape[0])
    np.random.shuffle(cnt)

    for i in cnt: 
        X_new[counter] = X[i]
        y_new[counter] = y[i]
        counter += 1

    # Check the shape of X before defining the model
    print("Shape of X:", X.shape)  # Debugging line
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")

    ip = Input(shape=(X.shape[1],))  # Ensure X has at least 2 dimensions

    m = Dense(512, activation="relu")(ip)
    m = Dense(256, activation="relu")(m)

    op = Dense(y.shape[1], activation="softmax")(m) 

    model = Model(inputs=ip, outputs=op)

    model.compile(optimizer='rmsprop', loss="categorical_crossentropy", metrics=['acc'])

    model.fit(X, y, epochs=50)

    model.save("model.h5")
    np.save("labels.npy", np.array(label))
else:
    print("No valid data loaded.")