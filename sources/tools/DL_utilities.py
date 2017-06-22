from __future__ import print_function

# from keras.models import Sequential, Model
# from keras.layers import Dense, Dropout, Activation, Flatten
# from keras.layers import Convolution2D, MaxPooling2D
# from keras.utils import np_utils
# from keras.models import load_model
# from keras.optimizers import SGD
# from keras import applications

import numpy as np
import csv

def initialize_headers_filepath(param_headers, filepath):
    with open(filepath, 'w') as csv_file:
        initialize_headers_file(param_headers, csv_file)
    return;

def initialize_headers_file(param_headers, csv_file):
    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(param_headers)
    return;

def log_history_to_csv_filepath(metaparams, training_perfs, filepath):
    """ Write the perfs of a model into the given filepath, open and close the file at each call, use log_history_to_csv_file for cross-validation usage. """
    with open(filepath, 'wa') as csv_file:
        log_history_to_csv_file(metaparams, training_perfs, csv_file)
    return;


def log_history_to_csv_file(metaparams, history, csv_file):
    """ Write the perfs of a model into the given open file, the file needs to be already open. """
    csv_writer =  csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for i in range(len(history.history['loss'])):
        training_perfs = [i+1, history.history['loss'][i], history.history['acc'][i], history.history['val_loss'][i], history.history['val_acc'][i]]
        csv_writer.writerow(metaparams + training_perfs)
    return;

