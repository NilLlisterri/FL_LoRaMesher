from node_manager import NodeManager
from dotenv import dotenv_values
import os
import time
from constants import *
import serial
import random

seed = 123

random.seed(123)

def main():
    devices = [
        # serial.Serial("com9", SERIAL_BR, timeout=5),
        # serial.Serial("com12", SERIAL_BR, timeout=5),
        # serial.Serial("com14", SERIAL_BR, timeout=5)
        serial.Serial("com5", SERIAL_BR, timeout=5),
        serial.Serial("com7", SERIAL_BR, timeout=5),
        serial.Serial("com11", SERIAL_BR, timeout=5)
    ]

    nodeManager = NodeManager(seed, devices)
    nodeManager.startExperiment()

if __name__ == "__main__":
    main()