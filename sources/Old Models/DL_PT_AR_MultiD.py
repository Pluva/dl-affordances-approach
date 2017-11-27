# DEPRECATED

# *---------------------------------------------*
# Multi Dataset Experiment
# Fork form the original MonoDataset experiment, in order to validate data on different
# split of the dataset. Should be use instead of MonoDataset from now on. Even for a
# small number of dataset (shouldn't be less than 5 however, to avoid perfect/worst case).
# *---------------------------------------------*


# ---------------------------------
# Keras and TF backend use GPU by default, uncoment this section of code in order to force CPU usage.
# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
# ---------------------------------

#python
from __future__ import print_function
from itertools import product
from time import time
import subprocess

#keras
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.models import load_model
from keras import optimizers
from keras import applications

#local
from tools.DL_models import *
from tools.DL_utilities import *





batch_size = 32
nb_classes = 2

# input image dimensions
# img_rows, img_cols = 124, 124
img_rows, img_cols = 124, 124

# Images are RGB.
img_channels = 3

# --------------- Hard linked datasets
# path to dataset workstation / laptop
# dataset_path = '/home/luce_vayrac/kinect_datasets/DL_Ready/rollable_data_2'
# dataset_path = '/home/eze/kinect_datasets/DL_Ready/rollable_data_2/'
# paths for randomized datasets
dataset_path_source = '/home/luce_vayrac/kinect_datasets/DL_Ready/rollable_data_source_2c'
dataset_path_target = '/home/luce_vayrac/kinect_datasets/DL_Ready/rollable_data_randomized'

# ----- Define parameters for grid search
nb_epoch = 10

range_learning_rates = [0.001, 0.0001]
range_reset_layers = range(1) # Reseting layers imply to let them be trained, so adapt fine tuning accordingly. 
range_fine_tuning = range(1) # Number of layers to be trained, [-1] to train them all. 
nb_iteration_per_model = range(1)
nb_datasets = range(10)
range_layers = [
    # (1024, 1024), (1024, 512), (1024, 256), (1024, 128),
    # (512, 512), (512, 256), (512, 128),
    (256, 32), (256, 16), (256, 8),
    (128, 32), (128, 16), (128, 8)]
# range_layer1_size = [1024, 512, 256, 128]
# range_layer2_size = [1024, 512, 256, 128]

# ----- Define optimizer parameters    
# SGD
# optimizer = SGD(lr=learning_rate, decay=1e-2, momentum=0.9, nesterov=True)
decay_range=[0.0001]; momentum_range=[0.9]; nesterov=True;
# RMSprop
# optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
# rho=0.9; epsilon=1e-08; decay=0.0;

# ----- Saving logs to file params
save_logs = True
# workstation / laptop
log_file_path = '/home/luce_vayrac/python_ws/training logs/DL_PT_AR_Mult_VGG16_lighter124.csv'
# log_file_path = '/home/eze/python_ws/DL_pretrainedAffRoll_training_decay_high.log'
log_file = open(log_file_path, 'w')
log_file_param_headers = ['lr',
    'rl',
    'ft',
    'decay',
    'momentum',
    'dataset',
    'l1_size',
    'l2_size',
    'iter',
    'epoch',
    'loss',
    'acc',
    'val_loss',
    'val_acc']
initialize_headers_file(log_file_param_headers, log_file)

# ----- Bash Toolkits
def bash_command(cmd):
    p = subprocess.Popen(cmd, shell=True, executable='/bin/bash')
    p.wait()
    return;
def randomize_dataset(path_to_source, path_to_target):
    cmd = 'bash /home/luce_vayrac/bash_ws/create_mix_dataset.sh'+' '+ path_to_source +' '+path_to_target
    bash_command(cmd)
    return;
# -----

def load_dataset(path):
    # ------------------- DATA PREPROCESSING -------------------
    # This will do preprocessing and realtime data augmentation:
    # ----- Training dataset
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images
    train_generator = train_datagen.flow_from_directory(
        path+'/train',
        target_size=(img_cols,img_rows),
        batch_size=batch_size)

    # ----- Validation dataset
    test_datagen = ImageDataGenerator(
        rescale=1./255)
    validation_generator = test_datagen.flow_from_directory(
        path+'/validation',
        target_size=(img_cols,img_rows),
        batch_size=batch_size)
    return (train_generator, validation_generator);

# ----- Training function to include all corresponding code in a easier snippet of code to handle.
def train_model_basic(train_generator, validation_generator, optimizer, reset_layers, fine_tuning, nb_epoch):
    # ------------------- MODEL CREATION -------------------
    # --- MODEL VGG16
    base_model = generate_model_VGG16(include_top=False, weights='imagenet', input_shape=(img_rows, img_cols, img_channels))
    # --- MODEL CIFAR10
    # model = generate_model_CIFAR10(optimizer=optimizer, input_shape=(img_rows, img_cols, img_channels), nb_classes=nb_classes)
    # base_model = model
    # --- MODEL SORTING
    # model = load_model_SORTING('/home/luce_vayrac/python_ws/saved_model/dl_sorting_4')

    # print(base_model.summary())

    # --- Randomize top_layers
    if (reset_layers > 0):
        base_model = randomize_layers(reset_layers, base_model, model_type='Model')
        # Accord reseting layers and fine tuning
        if (reset_layers > fine_tuning) and (fine_tuning != -1):
            fine_tuning = reset_layers

    # --- VGG without top layers
    top_model = base_model.output
    top_model = (Flatten())(top_model)
    # top_model = (Dense(512, activation='relu'))(top_model)
    top_model = (Dense(global_l1_size, activation='relu'))(top_model) # to test out layers size
    top_model = (Dropout(0.5))(top_model)
    # top_model = (Dense(256, activation='relu'))(top_model)
    #top_model = (Dense(global_l2_size, activation='relu'))(top_model) # to test out layers size
    #top_model = (Dropout(0.5))(top_model)
    top_model = (Dense(nb_classes, activation='softmax'))(top_model)

    # --- VGG with top layers
    # base_model.layers.pop()
    # top_model = (Dense(nb_classes, activation='softmax'))(base_model.layers[-1].output)

    model = Model(inputs=base_model.input, outputs=top_model)

    # ------------------- FREEZING LAYERS -------------------
    # Freeze layers
    if fine_tuning >= 0:
        layers_to_freeze = max(0, len(base_model.layers) - fine_tuning)
        for layer in base_model.layers[:layers_to_freeze]:
            layer.trainable = False

    # ------------------- COMPILING -------------------
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # ------------------- TRAINING -------------------
    # Fit the model on the batches generated by datagen.flow().
    history = model.fit_generator(generator=train_generator,
        # steps_per_epoch=5,
        # steps_per_epoch=train_generator.samples,
        steps_per_epoch=train_generator.samples / batch_size,
        epochs=nb_epoch,
        verbose=1,
        validation_data=validation_generator,
        # validation_steps=5,
        validation_steps=validation_generator.samples / batch_size,
        # class_weight=None, max_q_size=10, workers=1, pickle_safe=False, initial_epoch=0
    )

    return history;




# ----------- TRAINING

# ----- Progress var
progress = 0
grid_size=1.0
for n in [range_learning_rates,
    range_reset_layers,
    range_fine_tuning,
    decay_range,
    momentum_range,
    nb_datasets,
    range_layers,
    nb_iteration_per_model]:
    grid_size *= len(n)

starting_time = time()

# Outloop vars
current_dataset = -1

# ----- Training loops
print('-- Starting training:')
for dataset, global_layers_size, learning_rate, reset_layers, fine_tuning, decay, momentum, it in product(
                                                        nb_datasets,
                                                        range_layers,
                                                        range_learning_rates,
                                                        range_reset_layers,
                                                        range_fine_tuning,
                                                        decay_range,
                                                        momentum_range,
                                                        nb_iteration_per_model):

    # --- Set global layers size
    (global_l1_size, global_l2_size) = global_layers_size

    # --- Print progress info
    print('-- lr='+str(learning_rate)
        +' rl='+str(reset_layers)
        +' ft='+str(fine_tuning)
        +' decay='+str(decay)
        +' dataset='+str(dataset)
        +' l1_size='+str(global_l1_size)
        +' l2_size='+str(global_l2_size)
        +' it='+str(it+1)
        +' progress='+str(progress/grid_size))
    if progress >= 1:
        time_remaining = ((time() - starting_time) * (grid_size/progress) ) *(1-progress/grid_size) / 60
        print('-- -- remaining time= ' + str(time_remaining))
    progress += 1


    # --- Randomize and load dataset !! EXTERNALIZE THIS IF SAME DATASET !!
    if current_dataset != dataset:
        print('-- -- Randomizing Dataset')
        current_dataset = dataset
        randomize_dataset(dataset_path_source, dataset_path_target)
        (td, vd) = load_dataset(dataset_path_target)

    # --- Build optimizer
    optimizer = SGD(lr=learning_rate, decay=decay, momentum=momentum, nesterov=nesterov)
    # --- Train model
    history = train_model_basic(td, vd, optimizer, reset_layers, fine_tuning, nb_epoch)
    # --- Log results
    if save_logs:
        metaparams = [learning_rate, reset_layers, fine_tuning, decay, momentum, dataset, global_l1_size, global_l2_size, it]
        # training_perfs = [history.history['loss'][-1], history.history['acc'][-1], history.history['val_loss'][-1], history.history['val_acc'][-1]]
        log_history_to_csv_file(metaparams, history, log_file)
    # --- Clear TF session to counter memory leaks
    K.clear_session()
    
            

# ------------------- MODEL SAVING -------------------

    # # save model
    # if _save_model:
    #     print("Saving model after " + str(i * 50) + " epochs")
    #     # model.save("" + str(i*50) + "e.h5")

    # i += 1
