from node_manager import NodeManager
from dotenv import dotenv_values
import os
import time
from constants import *
import serial


def main():
    devices = [
        serial.Serial("com4", SERIAL_BR, timeout=5),
        # serial.Serial("com6", SERIAL_BR, timeout=5)
    ]

    nodeManager = NodeManager(devices)
    nodeManager.startExperiment()

if __name__ == "__main__":
    main()