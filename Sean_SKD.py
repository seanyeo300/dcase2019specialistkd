#Import Packages
import tensorflow as tf
from tensorflow.python.platform import build_info
import numpy as np
import pandas as pd
import pathlib
import os
import librosa
import librosa.display
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime 
from useful_functions import *

tensorflow.config.list_physical_devices('GPU')

filedir = 'D:/Sean/DCASE/datasets/Extract_to_Folder/TAU-urban-acoustic-scenes-2022-mobile-development/audio/'

sr = 44100


print('done')
