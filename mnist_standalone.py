import tensorflow as tf
from keras import backend as K
from keras.datasets import mnist
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Dense, Input, Flatten
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, TensorBoard
from clusterone import get_data_path, get_logs_path

log_dir = get_logs_path('/Users/artem/Documents/Scratch/mnist_keras_distributed/logs/')

def train():
    #
    # Data
    #

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    y_train = to_categorical(y_train, num_classes = 10)
    y_test = to_categorical(y_test, num_classes = 10)
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    #
    # Model
    #

    img_inp = Input(shape = (28, 28))
    x = Flatten()(img_inp)
    x = Dense(128, activation = 'relu')(x)
    x = Dense(128, activation = 'relu')(x)
    preds = Dense(10, activation = 'softmax')(x)

    model = Model(img_inp, preds)
    model.compile(
        optimizer = Adam(),
        loss = 'categorical_crossentropy',
        metrics = ['accuracy']
    )

    #
    # Callbacks
    #

    checkpoint = ModelCheckpoint(
        filepath = './weights.{epoch:02d}-{val_loss:.2f}.h5',
        monitor = 'val_acc',
        verbose = 1,
        save_best_only = True
    )

    tensorboard = TensorBoard(log_dir = log_dir)

    callbacks = [checkpoint, tensorboard]

    #
    # Train
    #

    model.fit(
        x_train,
        y_train,
        batch_size = 128,
        epochs = 10,
        validation_data = (x_test, y_test),
        callbacks = callbacks
    )

if __name__ == '__main__':
    train()


