import timeit

setup = """
import pickle
import os
import tensorflow as tf
from tensorflow import keras

curr_dir = os.getcwd()
pickles = "/cfg/data.pickle"
activationfunc = "relu"
weight_init = "he_uniform"
def create_model(training, output):

    net = keras.Sequential([
        keras.layers.Dense(len(training[0]), kernel_initializer = weight_init),
        keras.layers.Dense(128, kernel_initializer = weight_init, activation = activationfunc),
        keras.layers.Dense(64, kernel_initializer = weight_init, activation = activationfunc),
        keras.layers.Dense(64, kernel_initializer = weight_init, activation = activationfunc),
        keras.layers.Dense(32, kernel_initializer = weight_init, activation = activationfunc),
        keras.layers.Dense(24, kernel_initializer = weight_init, activation = activationfunc),
        keras.layers.Dense(len(output[0]), kernel_initializer = weight_init, activation='softmax') 
    ])

    net.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return net

with open(curr_dir + pickles, "rb") as f:
    words, labels, training, output = pickle.load(f)

with open(curr_dir + "/evaluations2.txt", "a+") as file:
    file.write("\\n")
    file.write(activationfunc + " " + weight_init + "5 Hidden layers, 24 neurons last")
    file.write("\\n")
"""

print(timeit.timeit("""model = create_model(training, output)
model.fit(training, output, epochs = 100, verbose = 1)
loss, accuracy = model.evaluate(training, output)
with open(curr_dir + "/evaluations2.txt", "a+") as file:
    file.write("loss " + str(loss))
    file.write("\\n")
    file.write("acc " + str(accuracy))
    file.write("\\n")
""", setup = setup, number=1))