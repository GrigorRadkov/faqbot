import tensorflow as tf
from tensorflow import keras

models = r"D:\Projects\faqbot\cfg\model.tf"

def create_model(training, output):

    net = keras.Sequential([
        keras.layers.Dense(len(training[0])),
        keras.layers.Dense(8, activation='relu'),
        keras.layers.Dense(8, activation='relu'),
        keras.layers.Dense(len(output[0]), activation='softmax') 
    ])

    net.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    net.fit(training, output, batch_size = 8, epochs = 1000)
    net.save(models, overwrite=False)

    print("Model saved")

    loss, accuracy = net.evaluate(training, output)
    
    print(" Loss = ", loss)
    print(" Accuracy = ", accuracy)
    print(net.summary())

    return net

def load_model(training, output):
    
    net = keras.models.load_model(models)
    
    print("Model loaded")
    print(net.summary())

    return net