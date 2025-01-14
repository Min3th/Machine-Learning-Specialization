import numpy as np 
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 
from IPython.display import display,Markdown,Latex
from sklearn.datasets import make_blobs
from matplotlib.widgets import Slider
from lab_utils_common import dlc 
from lab_utils_softmax import plt_softmax
import logging 
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

def my_softmax(z): # z is an array
  ez = np.exp(z)
  sm = ez/np.sum(ez) #sm is an array 
  return(sm)

plt.close("all")
plt_softmax(my_softmax)