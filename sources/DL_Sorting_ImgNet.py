# *---------------------------------------------*
# Sorting experiment on Imagenet Datasets.
# Use this code to pretrain a network.
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




# input image dimensions
# img_rows, img_cols = 128, 128
img_rows, img_cols = 224, 224

# Images are RGB.
img_channels = 3

# ----- Model Type
model_type = 'VGG16'
model_params = []

# --------------- Hard linked datasets
# path_to_dataset = '/home/luce_vayrac/Imagenet_datasets/Tools/' # Tools
# nb_classes = 7 # Tools
# path_to_dataset = '/home/luce_vayrac/Imagenet_datasets/Simple_Shapes/' # Shapes
# nb_classes = 6 # Shapes
path_to_dataset = '/home/luce_vayrac/Imagenet_datasets/Flowers/' # Shapes
nb_classes = 9 # Shapes

# ----- Define parameters for grid search
nb_epoch = 20

range_learning_rates = [0.001]
range_reset_layers = [0] # Reseting layers imply to let them be trained, so adapt fine tuning accordingly. 
range_fine_tuning = [-1] # Number of layers to be trained, [-1] to train them all. 
nb_iteration_per_model = range(1)

last_layer_sizes = [128]

# ----- Define optimizer parameters    
# SGD
# optimizer = SGD(lr=learning_rate, decay=1e-2, momentum=0.9, nesterov=True)
decay_range=[0.0001]; momentum_range=[0.9]; nesterov=True;
# RMSprop
# optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
# rho=0.9; epsilon=1e-08; decay=0.0;

# ----- Saving logs training history and model's weights
save_logs = True
save_model = True
logs_file_path_root = '/home/luce_vayrac/python_ws/saved_models/dl_vgg16/DL_SORTING_IMGNET_Flowers_VGG16_d'
model_file_path_root = '/home/luce_vayrac/python_ws/saved_models/dl_vgg16/DL_SORTING_IMGNET_Flowers_VGG16_d'


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
def train_model(model, train_generator, validation_generator, optimizer, reset_layers, fine_tuning, nb_epoch):
    """
    Train the given model with the given parameters. 
    """
    # --- Randomize top_layers
    if (reset_layers > 0):
        shuffle_weights(model, reset_layers)
        # Accord reseting layers and fine tuning
        if (reset_layers > fine_tuning) and (fine_tuning != -1):
            fine_tuning = reset_layers

    # ------------------- FREEZING LAYERS -------------------
    # Freeze layers
    if fine_tuning >= 0:
        layers_to_freeze = max(0, len(model.layers) - fine_tuning)
        for layer in model.layers[:layers_to_freeze]:
            layer.trainable = False

    # ------------------- COMPILING -------------------
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # ------------------- TRAINING -------------------
    # Fit the model on the batches generated by datagen.flow().
    history = model.fit_generator(generator=train_generator,
        # steps_per_epoch=5,-
        # steps_per_epoch=train_generator.samples,
        steps_per_epoch=train_generator.samples / batch_size,
        epochs=nb_epoch,
        verbose=1,
        validation_data=validation_generator,
        # validation_steps=5,
        validation_steps=validation_generator.samples / batch_size,
        # class_weight=None, max_q_size=10, workers=1, pickle_safe=False, initial_epoch=0
    )
    return (history, model);

def construct_model(model_type, params):
    # ------------------- MODEL CREATION -------------------
    if model_type == 'VGG16':
        # --- MODEL VGG16
        # params[dense layer size, ]
        base_model = generate_model_VGG16(include_top=False, weights='imagenet', input_shape=(img_rows, img_cols, img_channels))
        x = base_model.output
        x = (Flatten())(x)
        if (params[0] != 0):
            x = (Dense(params[0], activation='relu'))(x) # to test out layers size
        x = (Dropout(0.5))(x)
        # x = (Dense(256, activation='relu'))(x)
        #x = (Dropout(0.5))(x)
        x = (Dense(nb_classes, activation='softmax'))(x)

        model = Model(inputs=base_model.input, outputs=x)

    elif model_type == 'VGG19':
        # --- MODEL VGG19
        # params[dense layer size, ]
        base_model = generate_model_VGG19(include_top=False, weights='imagenet', input_shape=(img_rows, img_cols, img_channels))
        x = base_model.output
        x = (Flatten())(x)
        x = (Dense(params[0], activation='relu'))(x) # to test out layers size
        x = (Dropout(0.5))(x)
        # x = (Dense(256, activation='relu'))(x)
        #x = (Dropout(0.5))(x)
        x = (Dense(nb_classes, activation='softmax'))(x)

        model = Model(inputs=base_model.input, outputs=x)

    elif model_type == 'CIFAR10':
        # --- MODEL CIFAR10
        # model = generate_model_CIFAR10(optimizer=optimizer, input_shape=(img_rows, img_cols, img_channels), nb_classes=nb_classes)
        # base_model = model
        base_model = load_model_CIFAR10('/home/luce_vayrac/python_ws/saved_models/dl_cifar10/dl_cifar10_200_sgd.h5', include_top=False, top_size=6)

        x = base_model.output
        x = Flatten(name='top_flat1')(x)
        x = Dense(256, name='top_d1')(x)
        x = Activation('relu', name='top_a1')(x)
        x = Dropout(0.5, name='top_drop1')(x)
        x = Dense(nb_classes, name="top_d2")(x)
        x = Activation('softmax', name='top_a2')(x)

        model = Model(inputs=base_model.input, outputs=x)

    elif model_type == 'SORTING':
        # --- MODEL SORTING
        model = generate_model_SORTING(nb_classes, optimizer)


    return model;


# ------------------- TRAINING -------------------
logs_file_param_headers = ['lr',
            'rl',
            'ft',
            'decay',
            'momentum',
            'iter',
            'epoch',
            'loss',
            'acc',
            'val_loss',
            'val_acc']

# --- Progress var
progress = 0
grid_size=1.0
for n in [range_learning_rates,
    range_reset_layers,
    range_fine_tuning,
    decay_range,
    momentum_range,
    nb_iteration_per_model]:
    grid_size *= len(n)

starting_time = time()

# --- Outloop vars
current_dataset = -1

# --- Training loops
print('-- Starting training:')
for learning_rate, reset_layers, fine_tuning, decay, momentum, it, last_layer_size in product(
                                                        range_learning_rates, 
                                                        range_reset_layers,
                                                        range_fine_tuning,
                                                        decay_range,
                                                        momentum_range,
                                                        nb_iteration_per_model,
                                                        last_layer_sizes):


    # --- Print progress info
    print('-- lr='+str(learning_rate)
        +' rl='+str(reset_layers)
        +' ft='+str(fine_tuning)
        +' decay='+str(decay)
        +' it='+str(it+1)
        +' progress='+str(progress/grid_size))
    if progress >= 1:
        time_remaining = ((time() - starting_time) * (grid_size/progress) ) *(1-progress/grid_size) / 60
        print('-- -- remaining time= ' + str(time_remaining))
    progress += 1

    (td, vd) = load_dataset(path_to_dataset)

    # --- Construct model
    model_params = [last_layer_size]
    model = construct_model(model_type, model_params)
    # print(model.summary())
    # --- Build optimizer
    optimizer= SGD(lr=learning_rate, decay=decay, momentum=momentum, nesterov=nesterov)
    # --- Train model
    (history, model) = train_model(model, td, vd, optimizer, reset_layers, fine_tuning, nb_epoch)
    # --- Log results
    logs_input_size = str(img_cols) + 'x' + str(img_rows)
    if save_logs:
        metaparams = [learning_rate, reset_layers, fine_tuning, decay, momentum, it]
        logs_file_path = logs_file_path_root + str(last_layer_size) + '_' + logs_input_size + '.csv'
        logs_file = open(logs_file_path, 'w')
        initialize_headers_file(logs_file_param_headers, logs_file)
        log_history_to_csv_file(metaparams, history, logs_file)

    if save_model:
        model_file_path = model_file_path_root + str(last_layer_size) + '_' + logs_input_size + '.h5'
        model.save(model_file_path)

    # --- Clear TF session to counter memory leaks
    K.clear_session()
    
            


