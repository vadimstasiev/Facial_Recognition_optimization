import sys
from os import listdir
from matplotlib import image
from matplotlib import pyplot
import numpy as np
from numpy import asarray
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import cv2
import matplotlib.pyplot as plt
from tensorboard.plugins.hparams import api as hp

tf.get_logger().setLevel('INFO')
PATH = "."
DNAME = "/dataset"
FILES = listdir(PATH + DNAME)
N_FILES = len(FILES)

IM1 = image.imread(PATH + DNAME + '/' + FILES[0])
N_PIXELS = IM1.shape[0]*IM1.shape[1]

# HYPER PARAMETERS
HP_EPOCHS = hp.HParam('epochs', hp.IntInterval([12, 30]))
HP_NEURONS = hp.HParam('num_units', hp.IntInterval([50, 240]))
HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1, 0.2))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd', 'nadam']))

METRIC_ACCURACY = 'accuracy'

with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
  hp.hparams_config(
    hparams=[HP_EPOCHS, HP_NEURONS, HP_DROPOUT, HP_OPTIMIZER],
    metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
  )

def get_parsed_dataset():
    images = []
    labels = []
    for i in range(0, N_FILES):
        filename = FILES[i]
        img = cv2.imread(PATH + DNAME + '/' + filename, cv2.IMREAD_GRAYSCALE)
        images.append(img)
        lbl = int(filename[5:7])
        labels.append(lbl)
    temp_labels = labels.copy()
    for n in range(len(set(labels))-1, -1, -1):
        current_max = max(temp_labels)
        for i in range(len(labels)-1, -1, -1):
            if temp_labels[i] == current_max:
                temp_labels[i] = -1
                labels[i] = n
    return (images, labels)


def create_model(POSSIBLE_OUTPUT, hparams):
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(
            IM1.shape[0], IM1.shape[1])),  # width and height
        keras.layers.Dense(hparams[HP_NEURONS], activation='relu'),
        keras.layers.Dropout(hparams[HP_DROPOUT]),
        keras.layers.Dense(POSSIBLE_OUTPUT)  # (30) Possible Output Options
    ])
    model.compile(optimizer=hparams[HP_OPTIMIZER],
                  loss=keras.losses.SparseCategoricalCrossentropy(
                      from_logits=True),
                  metrics=['accuracy'])
    return model


def train_model(model, train_data, train_labels, hparams):
    model.fit(x=np.array(train_data),
              y=np.array(train_labels),
              epochs=hparams[HP_EPOCHS],
              callbacks=[
                tf.keras.callbacks.TensorBoard(logdir),  # log metrics
                hp.KerasCallback(logdir, hparams),  # log hparams
              ]
    )
    return model


def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    plt.xlabel("{} {:2.0f}% ({})".format([predicted_label],
                                         100*np.max(predictions_array),
                                         [true_label]),
               color=color)


def plot_value_array(predictions_array):
    plt.grid(False)
    x_range = range(len(predictions_array))
    plt.xticks(x_range)
    plt.yticks([])
    thisplot = plt.bar(x_range,
                       predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color('red')


if __name__ == "__main__":
    dataset_images, dataset_labels = get_parsed_dataset()
    dataset_images = np.array(dataset_images)
    dataset_images = dataset_images / 255.0
    # X represents the dataset_images and Y represents the dataset_labels
    X_train, X_test, Y_train, Y_test = train_test_split(
        dataset_images, dataset_labels, test_size=0.3, random_state=42)
    POSSIBLE_OUTPUT = len(set(dataset_labels))  # 30
    model = train_model(create_model(POSSIBLE_OUTPUT, hparams), X_train, Y_train, hparams)
    test_loss, test_acc = model.evaluate(
        np.array(X_test),  np.array(Y_test), verbose=2)
    print('\nTest accuracy:', test_acc)
    probability_model = tf.keras.Sequential([model,
                                             tf.keras.layers.Softmax()])
    predictions = probability_model.predict(np.array(X_train))
    num_rows = 5
    num_cols = 2
    num_images = num_rows*num_cols
    plt.rcParams.update({'font.size': 7})
    plt.figure(figsize=(12*num_cols, 2*num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2*num_cols, 2*i+1)
        plot_image(i, predictions[i], Y_test, np.array(X_test))
        plt.subplot(num_rows, 2*num_cols, 2*i+2)
        plot_value_array(predictions[i])
    plt.tight_layout()
    plt.show()


    ######################################
    # session_num = 0

    # for num_units in HP_NUM_UNITS.domain.values:
    # for dropout_rate in (HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value):
    #     for optimizer in HP_OPTIMIZER.domain.values:
    #     hparams = {
    #         HP_NUM_UNITS: num_units,
    #         HP_DROPOUT: dropout_rate,
    #         HP_OPTIMIZER: optimizer,
    #     }
    #     run_name = "run-%d" % session_num
    #     print('--- Starting trial: %s' % run_name)
    #     print({h.name: hparams[h] for h in hparams})
    #     run('logs/hparam_tuning/' + run_name, hparams)
    #     session_num += 1