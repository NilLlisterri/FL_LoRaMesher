from node_manager import NodeManager
from dotenv import dotenv_values
import os
import time
from constants import *
import serial
import random

seed = 1234

random.seed(1234)

def main():
    devices = [
        #serial.Serial("com5", SERIAL_BR, timeout=15),
        serial.Serial("com7", SERIAL_BR, timeout=15),
        serial.Serial("com11", SERIAL_BR, timeout=15)
    ]

    device_address_map = {
        #"com5": 1, #57156,# 22240,
        "com7": 2, #2120, # 2120,
        "com11": 3, #56640, # 22652,
    }

    nodeManager = NodeManager(seed, devices, device_address_map)
    nodeManager.startExperiment()

if __name__ == "__main__":
    main()