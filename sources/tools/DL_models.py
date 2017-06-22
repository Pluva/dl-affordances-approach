from __future__ import print_function
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.layers import convolutional, MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.utils import np_utils
from keras.models import load_model
from keras.optimizers import SGD
from keras import applications

import numpy as np

# Global constant relating to predefined models
DL_MODEL_SORTING = 0
DL_MODEL_CIFAR10 = 1
DL_MODEL_VGG16 = 2
DL_MODEL_VGG19 = 3

def generate_model_SORTING(nb_classes, input_shape=(124,124,3)):
    """ Create and returns a compiled model for the sorting task. """

    i = Input(input_shape)
    
    x = Conv2D(32, 3, 3, border_mode='same')(i)
    x = Activation('relu')(x)
    x = Conv2D(32, 3, 3)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(64, 3, 3, border_mode='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(64, 3, 3)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Flatten()(x)
    x = Dense(256)(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(nb_classes)(x)
    x = Activation('softmax')(x)

    # # Default optimizer
    # if optimizer==None:
    #     optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    model = Model(inputs=i, outputs=x)
    # model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model;

def load_model_SORTING(model_path, include_top, top_size=2):
    model = load_model(model_path)
    # Remove top
    if (include_top == False):
        model = Model(inputs=model.input, outputs=model.layers[-top_size].output)
    

    return model;

def generate_model_CIFAR10(nb_classes, input_shape=(128,128,3)):
    """ Create and returns a compiled 'CIFAR10' model. """

    i = Input(input_shape)
    
    x = Conv2D(32, 3, 3, border_mode='same')(i)
    x = Activation('relu')(x)
    x = Conv2D(32, 3, 3)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(64, 3, 3, border_mode='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(64, 3, 3)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Flatten()(x)
    x = Dense(256)(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(nb_classes)(x)
    x = Activation('softmax')(x)

    # # Default optimizer
    # if optimizer==None:
    #     optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    model = Model(inputs=i, outputs=x)
    # model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model;

def load_model_CIFAR10(model_path, include_top, top_size=2):
    model = load_model(model_path)
    # Remove top
    if (include_top == False):
        model = Model(inputs=model.input, outputs=model.layers[-top_size-1].output)
    return model;

def generate_model_VGG16(include_top=True, weights='imagenet', input_tensor=None, input_shape=None):
    """ Generate and returns the predifined VGG16 network from keras. """

    # model = Sequential()
    # model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    # model.add(Convolution2D(64, 3, 3, activation='relu'))
    # model.add(ZeroPadding2D((1,1)))
    # model.add(Convolution2D(64, 3, 3, activation='relu'))
    # model.add(MaxPooling2D((2,2), strides=(2,2)))

    # model.add(ZeroPadding2D((1,1)))
    # model.add(Convolution2D(128, 3, 3, activation='relu'))
    # model.add(ZeroPadding2D((1,1)))
    # model.add(Convolution2D(128, 3, 3, activation='relu'))
    # model.add(MaxPooling2D((2,2), strides=(2,2)))

    # model.add(ZeroPadding2D((1,1)))
    # model.add(Convolution2D(256, 3, 3, activation='relu'))
    # model.add(ZeroPadding2D((1,1)))
    # model.add(Convolution2D(256, 3, 3, activation='relu'))
    # model.add(ZeroPadding2D((1,1)))
    # model.add(Convolution2D(256, 3, 3, activation='relu'))
    # model.add(MaxPooling2D((2,2), strides=(2,2)))

    # model.add(ZeroPadding2D((1,1)))
    # model.add(Convolution2D(512, 3, 3, activation='relu'))
    # model.add(ZeroPadding2D((1,1)))
    # model.add(Convolution2D(512, 3, 3, activation='relu'))
    # model.add(ZeroPadding2D((1,1)))
    # model.add(Convolution2D(512, 3, 3, activation='relu'))
    # model.add(MaxPooling2D((2,2), strides=(2,2)))

    # model.add(ZeroPadding2D((1,1)))
    # model.add(Convolution2D(512, 3, 3, activation='relu'))
    # model.add(ZeroPadding2D((1,1)))
    # model.add(Convolution2D(512, 3, 3, activation='relu'))
    # model.add(ZeroPadding2D((1,1)))
    # model.add(Convolution2D(512, 3, 3, activation='relu'))
    # model.add(MaxPooling2D((2,2), strides=(2,2)))

    # model.add(Flatten())
    # model.add(Dense(4096, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(4096, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(1000, activation='softmax'))

    model = applications.VGG16(include_top=include_top, weights=weights, input_tensor=input_tensor, input_shape=input_shape)
    # print(model.summary())
    return model;

def load_model_VGG16(model_path, include_top, top_size=2):
    model = load_model(model_path)
    # Remove top
    if (include_top == False):
        model = Model(inputs=model.input, outputs=model.layers[-top_size-1].output)
    return model;

def generate_model_VGG19(include_top=True, weights='imagenet', input_tensor=None, input_shape=None):
    """ Generate and returns the predifined VGG16 network from keras. """


    # if input_tensor is None:
    #     img_input = Input(shape=input_shape)
    # else:
    #     if not K.is_keras_tensor(input_tensor):
    #         img_input = Input(tensor=input_tensor, shape=input_shape)
    #     else:
    #         img_input = input_tensor
    # # Block 1
    # x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    # x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # # Block 2
    # x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    # x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # # Block 3
    # x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    # x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    # x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    # x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4')(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # # Block 4
    # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4')(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # # Block 5
    # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4')(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # if include_top:
    #     # Classification block
    #     x = Flatten(name='flatten')(x)
    #     x = Dense(4096, activation='relu', name='fc1')(x)
    #     x = Dense(4096, activation='relu', name='fc2')(x)
    #     x = Dense(classes, activation='softmax', name='predictions')(x)
    # else:
    #     if pooling == 'avg':
    #         x = GlobalAveragePooling2D()(x)
    #     elif pooling == 'max':
    #         x = GlobalMaxPooling2D()(x)

    model = applications.VGG19(include_top=include_top, weights=weights, input_tensor=input_tensor, input_shape=input_shape)
    return model;

def load_model_VGG19(model_path, include_top, top_size=2):
    model = load_model(model_path)
    # Remove top
    if (include_top == False):
        model = Model(inputs=model.input, outputs=model.layers[-top_size-1].output)
    return model;

def generate_model_RESNET50(include_top=True, weights='imagenet', input_tensor=None, input_shape=None):
    """ Generate and returns the predifined ResNet50 network from keras. """

    # See https://github.com/fchollet/keras/blob/master/keras/applications/resnet50.py
    # for more informations about the strucutre and construction of this network.
    model = resnet50.ResNet50(include_top=include_top, weights=weights, input_tensor=input_tensor, input_shape=input_shape)
    # print(model.summary())
    return model;

def load_model_RESNET50(model_path, include_top, top_size=2):
    model = load_model(model_path)
    # Remove top
    if (include_top == False):
        model = Model(inputs=model.input, outputs=model.layers[-top_size-1].output)
    return model;

def randomize_layers(nb_layers, old_model, model_type='Model'):
    """ Randomize the n top layers of a model.
    In lack of a better solution, for now this function generate a new model, 
    and then copy the weigts of the old model."""

    config = old_model.get_config()
    if model_type=='Model':
        new_model = Model.from_config(config)
    elif model_type=='Sequential':
        new_model = Sequential.from_config(config)
    else:
        print('Wrong parameter, model can only be Sequential or Model.')

    if nb_layers==-1:
        nb_layers = len(new_model.layers)
    else:
        nb_layers = min(nb_layers, len(new_model.layers))

    # Copy the weights of the non-randomized layers.
    for layer_i in range(len(new_model.layers) - nb_layers):
        new_model.layers[layer_i].set_weights(old_model.layers[layer_i].get_weights())

    del old_model

    return new_model;

# Shuffle weights method from @jkleint 'MANY THANKS'
def shuffle_weights(model, layers_to_shuffle, weights=None):
    """Randomly permute the weights in `model`, or the given `weights`.
    This is a fast approximation of re-initializing the weights of a model.
    Assumes weights are distributed independently of the dimensions of the weight tensors
      (i.e., the weights have the same distribution along each dimension).
    :param Model model: Modify the weights of the given model.
    :param list(ndarray) weights: The model's weights will be replaced by a random permutation of these weights.
    :param integer layers_to_shuffle: Number of layers to reinitialise starting from the top.
      If `None`, permute the model's current weights.
    """

    if weights is None:
        weights = model.get_weights()
    random_weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]

    if layers_to_shuffle >= 0:
        starting_layer = max(0, len(weights) - layers_to_shuffle - 1)
    else:
        starting_layer = 0

    for i in range(starting_layer, len(weights)):
        weights[i] = random_weights[i]

    model.set_weights(weights)

    return;

# randomize_layers(DL_MODEL_CIFAR10, 4)
# generate_model_CIFAR10()
# generate_model_VGG16()
# generate_model_RESNET50()
