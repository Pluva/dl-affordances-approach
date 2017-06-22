"""
Load a trained model and then perform prediction tasks over a set of objects.
"""

# ---------------------------------
# Keras and TF backend use GPU by default, uncoment this section of code in order to force CPU usage.
# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
# ---------------------------------

#python
from __future__ import print_function

#keras
from keras import backend as K

#local
from tools.DL_models import *
from tools.DL_utilities import *


