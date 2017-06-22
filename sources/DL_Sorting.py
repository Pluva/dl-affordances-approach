# *---------------------------------------------*
# Sorting Experiement
# Goal of this experiment is to see if the lower layers of the network can be trained to
# recognize objects based on very simple differences. Basically can the network build
# features distinctive enough to let the top layers separate objects whithout classes.
# With the ending goal being to then transfer those layers into more complex tasks.
# *---------------------------------------------*

# -----------------
# Keras and TF backend use GPU by default, uncoment this section of code in order to force CPU usage.
# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
# -----------------

#python
from __future__ import print_function
from itertools import product
from time import time
import subprocess

#keras
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.utils import np_utils
from keras.models import load_model
from keras import optimizers
from keras import applications

#local
from tools.DL_models import *
from tools.DL_utilities import *


# BASH TOOLKIT
def bash_command(cmd):
    p = subprocess.Popen(cmd, shell=True, executable='/bin/bash')
    p.wait()
    return;
def randomize_dataset_sorting(nb_objects, path_to_source, path_to_target):
    cmd = 'bash /home/luce_vayrac/bash_ws/create_crible_dataset.sh' + ' ' + str(nb_objects) +' '+ path_to_source +' '+path_to_target
    bash_command(cmd)
    return;
# ------------



# input image dimensions
img_rows, img_cols = 124, 124
img_channels = 3

# ----- Number of classes (in this case, the number of objects picked up).
nb_classes = 10

# -----
nb_epoch = 15

batch_size = 16

# ----- Define parameters for grid search
range_learning_rates = [0.001]
range_reset_layers = [4]
range_fine_tuning = [-1]
nb_sets_per_model = [50]
nb_iteration_per_model = range(1)

# ----- Define optimizer parameters    
# -- SGD
# optimizer = SGD(lr=learning_rate, decay=1e-2, momentum=0.9, nesterov=True)
decay_range=[0.0001];
momentum_range=[0.9];
nesterov=True;
# -- RMSprop
# optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
# rho=0.9; epsilon=1e-08; decay=0.0;

# ----- Saving logs to file params
save_logs = True
# workstation / laptop
log_file_path = '/home/luce_vayrac/python_ws/training logs/DL_Sorting10.csv'
# log_file_path = '/home/eze/python_ws/DL_pretrainedAffRoll_training_decay_high.log'
log_file = open(log_file_path, 'w')
log_file_param_headers = ['lr', 'rl', 'ft', 'decay', 'momentum', 'nb_datasets', 'iter', 'phase', 'epoch', 'loss', 'acc', 'val_loss', 'val_acc']
initialize_headers_file(log_file_param_headers, log_file)

# ----- Saving models
save_models = True
save_model_path = '/home/luce_vayrac/python_ws/saved_models/'

# --------------- Hard linked datasets
# path to dataset workstation / laptop
# dataset_path = '/home/luce_vayrac/kinect_datasets/DL_Ready/rollable_data_2'
# dataset_path = '/home/eze/kinect_datasets/DL_Ready/rollable_data_2/'
# paths for randomized datasets
dataset_path_source = '/home/luce_vayrac/kinect_datasets/DL_Ready/objects_data_source'
dataset_path_target = '/home/luce_vayrac/kinect_datasets/DL_Ready/objects_data_randomized'


def load_dataset(train_data, vali_data):
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
        vertical_flip=True)  # randomly flip images
    train_generator = train_datagen.flow_from_directory(
        train_data,
        target_size=(img_cols,img_rows),
        batch_size=batch_size)

    # ----- Validation dataset
    test_datagen = ImageDataGenerator(
        rescale=1./255)
    validation_generator = test_datagen.flow_from_directory(
        vali_data,
        target_size=(img_cols,img_rows),
        batch_size=batch_size)
    return (train_generator, validation_generator);


def train_model_sorting(model, train_generator, validation_generator, optimizer, reset_layers, fine_tuning, nb_epoch):
    """
    Train the sorting experiment model.
    """
    
    # Randomize top_layers
    if (reset_layers > 0):
        # model = randomize_layers(reset_layers, model, model_type='Model')
        shuffle_weights(model, reset_layers)
        # Accord reseting layers and fine tuning
        if (reset_layers > fine_tuning) and (fine_tuning != -1):
            fine_tuning = reset_layers

    
    # ------------------- FREEZING LAYERS -------------------
    # Freeze layers
    for layer in model.layers:
        layer.trainable = True
    if fine_tuning >= 0:
        layers_to_freeze = max(0, len(model.layers) - fine_tuning)
        for layer in model.layers[:layers_to_freeze]:
            layer.trainable = False

    # ------------------- COMPILING -------------------            
    # model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # ------------------- TRAINING -------------------
    # Fit the model on the batches generated by datagen.flow().
    history = model.fit_generator(generator=train_generator,
        # steps_per_epoch=5,
        steps_per_epoch=train_generator.samples / batch_size,
        epochs=nb_epoch,
        verbose=1,
        validation_data=validation_generator,
        # validation_steps=5,
        validation_steps=validation_generator.samples / batch_size,
        # class_weight=None, max_q_size=10, workers=1, pickle_safe=False, initial_epoch=0
    )

    return (history);



# ---------- TRAINING ----------
# ----- Progress var
progress = 0
grid_size=1.0
for n in [range_learning_rates,
    range_reset_layers,
    range_fine_tuning,
    decay_range,
    momentum_range,
    nb_sets_per_model,
    nb_iteration_per_model]:
    grid_size *= len(n)

starting_time = time()


# ----- Training loops
print('-- Starting training:')
for learning_rate, reset_layers, fine_tuning, decay, momentum, nb_sets, it in product(
                                                        range_learning_rates,
                                                        range_reset_layers,
                                                        range_fine_tuning,
                                                        decay_range,
                                                        momentum_range,
                                                        nb_sets_per_model,
                                                        nb_iteration_per_model):

    # Print progress info
    print('-- lr='+str(learning_rate)
        +' rl='+str(reset_layers)
        +' ft='+str(fine_tuning)
        +' decay='+str(decay)
        +' it='+str(it+1)
        +' nb_sets='+str(nb_sets)     
        +' progress='+str(progress/grid_size))
    if progress >= 1:
        time_remaining = ((time() - starting_time) * (grid_size/progress) ) *(1-progress/grid_size) / 60
        print('-- -- remaining time= ' + str(time_remaining))
    progress += 1

    # ------------------- OPTIMIZER -------------------
    optimizer_top = SGD(lr=learning_rate, decay=decay, momentum=momentum, nesterov=nesterov)
    optimizer_bot = SGD(lr=learning_rate/10, decay=decay, momentum=momentum, nesterov=nesterov)
    # ------------------- MODEL CREATION -------------------
    current_model = generate_model_SORTING(nb_classes=nb_classes, optimizer=optimizer_top, input_shape=(img_rows, img_cols, img_channels))
    # print(base_model.summary())
    # ------------------- COMPILING -------------------
    # current_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # ------------------- TRAINING -------------------
    oom_counter = 0
    for i in range(nb_sets):
        # OoM Workaround
        if oom_counter >= 20:
            current_model.save('/home/luce_vayrac/python_ws/tmp_model_save')
            del current_model
            K.clear_session()
            current_model = load_model('/home/luce_vayrac/python_ws/tmp_model_save')
            oom_counter = 0            
        oom_counter = oom_counter + 1

        # Randomize and load a new set of n objects
        randomize_dataset_sorting(nb_classes, dataset_path_source, dataset_path_target)
        (td, vd) = load_dataset(dataset_path_target, dataset_path_target)
        print('-- -- -- dataset ' + str(i))

        # Training top layers
        if i >= 1:
            (history1) = train_model_sorting(current_model, td, vd, optimizer_top, reset_layers, reset_layers, nb_epoch)

        # Fine tuning bottom layers
        (history2) = train_model_sorting(current_model, td, vd, optimizer_bot, 0, fine_tuning, nb_epoch)

        # Log results
        if save_logs:
            metaparams = [learning_rate, reset_layers, fine_tuning, decay, momentum, nb_sets, it]
            if i > 0:
                log_history_to_csv_file(metaparams + [0], history1, log_file)
            log_history_to_csv_file(metaparams + [1], history2, log_file)

    # Saving model
    if save_models:
        current_model.save(save_model_path + 'dl_sorting' + str(nb_classes) + '_' + str(reset_layers) + '.h5')

    # Clear TF session to counter memory leaks
    K.clear_session()
    
            

# ------------------- MODEL SAVING -------------------

    # # save model
    # if _save_model:
    #     print("Saving model after " + str(i * 50) + " epochs")
    #     # model.save("" + str(i*50) + "e.h5")

    # i += 1
