from ast import keyword
import warnings
from matplotlib import figure
from matplotlib.ticker import MaxNLocator
import serial
from serial.tools.list_ports import comports
import struct
import time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import threading
import time
import json
import os
import random
from queue import Queue
from scipy.io import wavfile
import sys

def readFloat(d):
    data = d.read(4)
    [float_num] = struct.unpack('f', data)
    return float_num

def readInt(d, size):
    return int.from_bytes(d.read(size), "little", signed=False)

def deScaleWeight(min_w, max_w, weight, scaledWeightBits):
    a, b = getScaleRange(scaledWeightBits)
    return min_w + ( (weight-a)*(max_w-min_w) / (b-a) )

def scaleWeight(min_w, max_w, weight, scaledWeightBits):
    a, b = getScaleRange(scaledWeightBits)
    return round(a + ( (weight-min_w)*(b-a) / (max_w-min_w) ))

def getScaleRange(scaledWeightBits):
    return 0, pow(2, scaledWeightBits)-1

def scaleWeights(weights, scaledWeightBits):
    min_w = min(weights)
    max_w = max(weights)
    return min_w, max_w, [scaleWeight(min_w, max_w, w, scaledWeightBits) for w in weights]

def deScaleWeights(min_w, max_w, weights, scaledWeightBits):
    return [deScaleWeight(min_w, max_w, w, scaledWeightBits) for w in weights]