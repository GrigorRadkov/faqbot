import tensorflow as tf
import tflearn as tfl

def create_model(training, output):

    tf.reset_default_graph()

    net = tfl.input_data(shape = [None , len(training[0])])
    net = tfl.fully_connected(net, 8)
    net = tfl.fully_connected(net, 8)
    net = tfl.fully_connected(net, len(output[0]), activation="softmax")
    net = tfl.regression(net)

    model = tfl.DNN(net)

    model.fit(training, output, n_epoch = 1000, batch_size = 8, show_metric= True)
    model.save("D:\Ot uni\Diplomna\FAQBot\cfg\model.tflearn")
    print("Model saved")
    return model

def load_model(training, output):

    tf.reset_default_graph()

    net = tfl.input_data(shape = [None , len(training[0])])
    net = tfl.fully_connected(net, 8)
    net = tfl.fully_connected(net, 8)
    net = tfl.fully_connected(net, len(output[0]), activation="softmax")
    net = tfl.regression(net)
    
    model = tfl.DNN(net)

    model.load("D:\Ot uni\Diplomna\FAQBot\cfg\model.tflearn")
    print("Model loaded")
    return model