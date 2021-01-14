import os
import tensorflow as tf
from tensorflow import keras

#Necessary paths
curr_dir = os.getcwd()
models = "\cfg\model.tf"

#Creates a keras deep neural network with ReLU functions in the hidden layers and a softmax at the end. 
#Fits the network by using the training and output data from the preprocessing phase. 
#Saves the model for later use in a model.tf folder. For more information on how to tinker with this please refer to the Tensorflow keras documentation.
def create_model(training, output):

    net = keras.Sequential([
        keras.layers.Dense(len(training[0])),
        keras.layers.Dense(8, activation='relu'),
        keras.layers.Dense(8, activation='relu'),
        keras.layers.Dense(len(output[0]), activation='softmax') 
    ])

    net.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    net.fit(training, output, batch_size = 8, epochs = 1000)
    net.save(curr_dir + models, overwrite=False)

    print("Model saved")

    loss, accuracy = net.evaluate(training, output)
    
    print(" Loss = ", loss)
    print(" Accuracy = ", accuracy)
    print(net.summary())

    return net

#Loads a preexisting model
def load_model(training, output):
    
    net = keras.models.load_model(curr_dir + models)
    
    print("Model loaded")
    print(net.summary())

    return net