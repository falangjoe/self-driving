import pickle
import numpy as np
import tensorflow as tf
from keras.layers import Input, Flatten, Dense
from keras.models import Model
# TODO: import Keras layers you need here

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('network', '', "Bottleneck features training file (.p)")
flags.DEFINE_string('dataset', '', "Bottleneck features validation file (.p)")


def load_bottleneck_data(network,dataset):
    """
    Utility function to load bottleneck features.

    Arguments:
        training_file - String
        validation_file - String
    """

    training_file = "./bottlenecks/" + network + "_" + dataset + "_100_bottleneck_features_train.p"
    validation_file = "./bottlenecks/" + network + "_" + dataset + "_bottleneck_features_validation.p"
    print('training_file',training_file)
    print('validation_file',validation_file)

    with open(training_file, 'rb') as f:
        train_data = pickle.load(f)
    with open(validation_file, 'rb') as f:
        validation_data = pickle.load(f)

    X_train = train_data['features']
    y_train = train_data['labels']
    X_val = validation_data['features']
    y_val = validation_data['labels']

    return X_train, y_train, X_val, y_val


def main(_):
    # load bottleneck data
    X_train, y_train, X_valid, y_valid = load_bottleneck_data(FLAGS.network, FLAGS.dataset)

    print(X_train.shape)
    print(y_train.shape)
    print(y_train[0:30])
    nb_classes = len(np.unique(y_train))
    y_train = np.squeeze(y_train)
    y_valid = np.squeeze(y_valid)
    
    EPOCHS = 15
    BATCH_SIZE = 128
    rate = 0.001
    
    # define model
    #input_shape = X_train.shape[1:]
    #inp = Input(shape=input_shape)
    #x = Flatten()(inp)
    #x = Dense(nb_classes, activation='softmax')(x)
    #model = Model(inp, x)
    #model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#
    ## train model
    #model.fit(X_train, y_train, BATCH_SIZE, EPOCHS, validation_data=(X_val, y_val), shuffle=True)
    
    x = tf.placeholder(tf.float32, (None,X_train.shape[1],X_train.shape[2],X_train.shape[3]))
    
    fc7 = tf.contrib.layers.flatten(x)
    print(fc7.get_shape().as_list())
    shape = (fc7.get_shape().as_list()[-1], nb_classes)  # use this shape for the weight matrix
    fc8W = tf.Variable(tf.truncated_normal(shape=shape, mean = 0, stddev = 0.01))
    fc8b = tf.Variable(tf.zeros(nb_classes))
    logits = tf.matmul(fc7, fc8W) + fc8b
    probs = tf.nn.softmax(logits)


    y = tf.placeholder(tf.uint8, (None))
    one_hot_y = tf.one_hot(y, nb_classes)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
    loss_operation = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate = rate)
    training_operation = optimizer.minimize(loss_operation)

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def evaluate(X_data, y_data):
        num_examples = len(X_data)
        total_accuracy = 0
        sess = tf.get_default_session()
        for offset in range(0, num_examples, BATCH_SIZE):
            batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
            accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
            total_accuracy += (accuracy * len(batch_x))
        return total_accuracy / num_examples
    
    # TODO: define your model and hyperparams here
    # make sure to adjust the number of classes based on
    # the dataset
    # 10 for cifar10
    # 43 for traffic

    # TODO: train your model here
    
    from sklearn.utils import shuffle

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        num_examples = len(X_train)
        
        print("Training...")
        print()
        for i in range(EPOCHS):
            X_train, y_train = shuffle(X_train, y_train)
            for offset in range(0, num_examples, BATCH_SIZE):
                end = offset + BATCH_SIZE
                batch_x, batch_y = X_train[offset:end], y_train[offset:end]
                sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
                
            validation_accuracy = evaluate(X_valid, y_valid)
            print("EPOCH {} ...".format(i+1))
            print("Validation Accuracy = {:.3f}".format(validation_accuracy))
            print()


# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
