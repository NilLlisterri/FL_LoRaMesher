from ast import keyword
import warnings
from matplotlib import figure
from matplotlib.ticker import MaxNLocator
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
import sys
from tqdm import tqdm
from tqdm import trange
from constants import *
import serial


class NodeManager:

    def __init__(self, seed, devices: list[serial.Serial], device_address_map: map):
        self.devices = devices
        self.device_address_map = device_address_map
        self.seed = seed

        self.samples_folder = "./datasets/keywords"
        train_samples_split = 160       # Number of samples for training of each keyword
        test_samples_split = 20         # Number of samples for training of each keyword

        # Experiment sizes
        self.training_epochs = 160      # Amount of training epochs. Can't be more than kws * train_samples_split
        self.testing_epochs = 0         # Amount of test samples of each keyword. Can't be more than kws * test_samples_split

        self.momentum = 0.9
        self.learningRate= 0.05

        self.enableTest = True
        self.enablePlot = False
        self.batchSize = 4             # Must be divisble by the amount of keywords

        self.keywords_buttons = {
            "montserrat": 1,
            "pedraforca": 2,
            "vermell": 3,
            "blau": 4,
        }

        self.experiment = 'iid'        # 'iid', 'no-iid', 'train-test', None
        self.debug = False
        self.useSerialModemPassthrough = True
        self.pauseListen = False       # So there are no threads reading the serial input at the same time

        self.graph = []
        self.fl_round_epochs = []


        # Load the dataset
        self.words = list(self.keywords_buttons.keys())
        files = []
        test_files = []
        for i, word in enumerate(self.words):
            file_list = os.listdir(f"{self.samples_folder}/{word}")
            if (len(file_list) < train_samples_split + test_samples_split): 
                sys.exit(f"[SERVER] Not enough samples for keyword {word}")
            random.shuffle(file_list)
            files.append(list(map(lambda f: f"{word}/{f}", file_list[0:train_samples_split])))
            test_files.append(list(map(lambda f: f"{word}/{f}", file_list[train_samples_split:(train_samples_split+test_samples_split)])))

        self.keywords = list(sum(zip(*files), ()))
        self.test_keywords = list(sum(zip(*test_files), ()))

        if (self.training_epochs > len(self.keywords) / len(self.devices)):
            sys.exit(f"[SERVER] Not enough training samples for {self.training_epochs} training epochs on {len(self.devices)} devices")
        if (self.testing_epochs > len(self.test_keywords)):
            sys.exit(f"[SERVER] Not enough testing samples for {self.testing_epochs} testing epochs")
        
        self.test_accuracies_map = {}
        self.test_errors_map = {}
        self.training_accuracy_map = {}
        self.training_errors_map = {}
        self.successes_map = {}          # Booleans

        self.min_weights_map = {}
        self.max_weights_map = {}

        for deviceIndex, device in enumerate(self.devices): 
            self.training_accuracy_map[deviceIndex] = []
            self.test_accuracies_map[deviceIndex] = []
            self.test_errors_map[deviceIndex] = []
            self.training_errors_map[deviceIndex] = [] # MSE errors
            self.successes_map[deviceIndex] = Queue() # Amount of right inferences
            self.min_weights_map[deviceIndex] = []
            self.max_weights_map[deviceIndex] = []

    # Send the blank model to all the devices
    def initDevices(self):
        threads = []
        for deviceIndex, device in enumerate(self.devices):
            hidden_layer = np.random.uniform(-0.5,0.5, SIZE_HIDDEN_LAYER).astype('float32')
            output_layer = np.random.uniform(-0.5, 0.5, SIZE_OUTPUT_LAYER).astype('float32')
            thread = threading.Thread(target=self.initDevice, args=(hidden_layer, output_layer, device))
            thread.daemon = True
            thread.start()
            threads.append(thread)
        for thread in threads: thread.join()

    def initDevice(self, hidden_layer, output_layer, device):
        if self.debug: print(f"[{device.port}] Initializing device...")
        device.reset_input_buffer()
        device.write(b'i')
        initConfirmation = device.readline().decode()
        if self.debug: print(f"[{device.port}] Init device confirmation:", initConfirmation)

        for i in trange(SIZE_HIDDEN_LAYER, desc="Sending hidden layer"):
            device.write(struct.pack('f', hidden_layer[i]))
        
        for i in trange(SIZE_OUTPUT_LAYER, desc="Sending output"):
            device.write(struct.pack('f', output_layer[i]))

        if self.debug: print(f"[{device.port}] Model sent")
        modelReceivedConfirmation = device.readline().decode()
        if self.debug: print(f"[{device.port}] Model received confirmation: {modelReceivedConfirmation}")

    # Batch size: The amount of samples to send
    def sendSamples(self, device, deviceIndex, batch_index):
        start = ((deviceIndex*self.training_epochs) + (batch_index * self.batchSize)) #%len(keywords)
        end = ((deviceIndex*self.training_epochs) + (batch_index * self.batchSize) + self.batchSize) #%len(keywords)

        if self.debug: print(f"[{device.port}] Sending samples of batch {batch_index + 1}, from {start+1} to {end}")

        if self.debug: print(f"[{device.port}] Sending train batch {batch_index + 1}...")
        for i in tqdm(range(start, end), desc=f"[{device.port}] Sending train batch {batch_index + 1}", ncols=0 if self.debug else None):
            filename = self.keywords[i % len(self.keywords)]
            keyword = filename.split("/")[0]
            num_button = self.keywords_buttons[keyword]

            if self.debug: print(f"[{device.port}] Sending sample {i+1} ({i+1-start}/{end-start})")
            error, success = self.sendSample(device, f"{self.samples_folder}/{filename}", num_button, deviceIndex)
            self.successes_map[deviceIndex].put(success)
            self.training_errors_map[deviceIndex].append(error)
            samplesAccuracyTick = sum(self.successes_map[deviceIndex].queue)/len(self.successes_map[deviceIndex].queue)
            # if debug: print(f"[{device.port}] Samples accuracy tick: {samplesAccuracyTick}")
            self.training_accuracy_map[deviceIndex].append(samplesAccuracyTick)

    def sendSample(self, device, samplePath, num_button, deviceIndex, only_forward = False):
        with open(samplePath) as f:
            if self.debug: print(f'[{device.port}] Sending train command')
            device.write(b't')
            # startConfirmation = device.readline().decode()
            # if self.debug: print(f"[{device.port}] Train start confirmation:", startConfirmation)

            device.write(struct.pack('B', num_button))
            button_confirmation = device.readline().decode() # Button confirmation
            if self.debug: print(f"[{device.port}] Button confirmation: {button_confirmation}")

            device.write(struct.pack('B', 1 if only_forward else 0))
            only_forward_conf = device.readline().decode()
            if self.debug: print(f"[{device.port}] Only forward confirmation: {only_forward_conf}") # Button confirmation

            data = json.load(f)
            values = data['payload']['values']
            
            for value in values:
                device.write(struct.pack('h', value))
                # device.read()

            sample_received_conf = device.readline().decode()
            if self.debug: print(f"[{device.port}] Sample received confirmation:", sample_received_conf)

            graphCommand = device.readline().decode()
            if self.debug: print(f"[{device.port}] Graph command received: {graphCommand}")
            error, num_button_predicted = self.read_graph(device, deviceIndex)

        return error, num_button == num_button_predicted
    
    def sendTestSamples(self, device, deviceIndex):
        errors_queue = Queue()
        successes_queue = Queue()
        
        if self.debug: print(f"[{device.port}] Sending test sample {self.testing_epochs}")
        for filename, index in tqdm(self.test_keywords[:self.testing_epochs], desc=f"[{device.port}] Sending test samples", ncols=0 if self.debug else None):
            if self.debug: print(f"[{device.port}] Sending test sample {index}")
            keyword = filename.split("/")[0]
            num_button = self.keywords_buttons[keyword]
            
            error, success = self.sendSample(device, f"{self.samples_folder}/{filename}", num_button, deviceIndex, True)
            errors_queue.put(error)
            successes_queue.put(success)

        test_accuracy = sum(successes_queue.queue)/len(successes_queue.queue)
        test_error = sum(errors_queue.queue)/len(errors_queue.queue)
        if self.debug: print(f"[{device.port}] Testing accuracy: {test_accuracy}")
        if self.debug: print(f"[{device.port}] Testing MSE: {test_error}")
        self.test_accuracies_map[deviceIndex].append(test_accuracy)
        self.test_errors_map[deviceIndex].append(test_error)

    def read_graph(self, device, deviceIndex):
        outputs = device.readline().decode().split()
        if self.debug: print(f'[{device.port}] Outputs: {outputs}')
        predicted_button = outputs.index(max(outputs))+1
        if self.debug: print(f'[{device.port}] Predicted button: {predicted_button}')
        error = float(device.readline().decode()[:-2])
        if self.debug: print(f"[{device.port}] Error: {error}")

        ne = device.readline()
        if self.debug: print(f"[{device.port}] Num epochs: {ne}")
        n_epooch = int(ne)

        nb = device.readline()[:-2]
        self.graph.append([n_epooch, error, deviceIndex])
        return error, outputs.index(max(outputs)) + 1

    def plot(self, title):
        warnings.filterwarnings("ignore")
        ax = plt.figure(figsize=(11, 5)).gca()
        # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        font_sm = 13
        font_md = 16
        font_xl = 18
        plt.rc('font', size=font_sm)          # controls default text sizes
        plt.rc('axes', titlesize=font_sm)     # fontsize of the axes title
        plt.rc('axes', labelsize=font_md)     # fontsize of the x and y labels
        plt.rc('xtick', labelsize=font_sm)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=font_sm)    # fontsize of the tick labels
        plt.rc('legend', fontsize=font_sm)    # legend fontsize
        plt.rc('figure', titlesize=font_xl)   # fontsize of the figure title

        colors = ['r', 'g', 'b', 'y']
        markers = ['-', '--', ':', '-.']
        
        plt.ion()
        plt.show(block=False)

        while True:
            plt.clf()

            epochs = 1
            for device_index, device in enumerate(self.devices):
                epoch = [x[0] for x in self.graph if x[2] == device_index]
                error = [x[1] for x in self.graph if x[2] == device_index]
                epochs = max(len(error), epochs)
                plt.plot(error, colors[device_index] + markers[device_index], label=f"Device {device_index}", marker='o')

            plt.legend()
            plt.xlim(left=0)
            plt.ylim(bottom=0, top=0.8)
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.title(title)
            plt.autoscale(axis='x')
            # plt.xticks(range(0, epochs))

            if (self.experiment == 'train-test'): plt.axvline(x=self.training_epochs)

            for epoch in self.fl_round_epochs:
                plt.axvline(epoch - 0.5, linestyle = 'dashed')

            plt.pause(0.1)
            time.sleep(0.4)
    
    def sendTestAllDevices(self):
        if (self.testing_epochs == 0): return
        threads = []
        for deviceIndex, device in enumerate(self.devices):
            thread = threading.Thread(target=self.sendTestSamples, args=(device, deviceIndex))
            thread.daemon = True
            thread.start()
            threads.append(thread)
        for thread in threads: thread.join()

    def plotAccuracies(self):
        plt.ylim(bottom=0, top=1)
        plt.xlim(left=0)
        plt.autoscale(axis='x')
        colors = ['r', 'g', 'b', 'y']
        markers = ['-', '--', ':', '-.']
        for device_index, device in enumerate(self.devices):
            plt.plot(self.test_accuracies_map[device_index], colors[device_index] + markers[device_index], label=f"Device {device.port}", marker='o')
        plt.legend()

    # Trigger a FL round on a device. A target device can be specified
    def doFL(self, device: serial.Serial, target_device: serial.Serial = None):
        print(f"[SERVER] Triggering FL round on device {device.port}")
        device.write(b'>')
        device.write(struct.pack('H', self.device_address_map[target_device.port] if target_device != None else 0))

        fl_start_confirmation = device.readline().decode()
        if self.debug: print(f"[{device.port}] Fl start confirmation: {fl_start_confirmation}")

        # The device will request the routing table
        if self.useSerialModemPassthrough: self.answerRoutingTable(device)
        
        nodes_count = device.readline().decode()
        
        # The device will ask all the other devices for metrics
        if self.useSerialModemPassthrough: self.relayModemMessage(device, True)
        
        if self.debug: print(f"[{device.port}] Routing nodes count: {nodes_count}")
        if (nodes_count == "0\r\n"):
            print("No nodes found")
            exit()

        max_epochs_since_last_fl = device.readline().decode()
        if self.debug: print(f"[{device.port}] Max epochs since last FL: {max_epochs_since_last_fl}")
        if (max_epochs_since_last_fl == "0\r\n"):
            print("No new samples on other nodes")
            exit()
        
        localWeightFactor = device.readline().decode()
        if self.debug: print(f"[{device.port}] Local weights factor: {localWeightFactor}")
        externalWeightFactor = device.readline().decode()
        if self.debug: print(f"[{device.port}] External weights factor: {externalWeightFactor}")
        numBatches = device.readline().decode()
        if self.debug: print(f"[{device.port}] Num batches: {numBatches}")

        if self.useSerialModemPassthrough: 
            for i in trange(int(numBatches), desc="Transfering batch"):
                batchRequestMessage = device.readline().decode()
                if self.debug: print(f"[{device.port}] BatchRequestMessage: {batchRequestMessage}")
                # The device will ask for the weight batches
                self.relayModemMessage(device, True)
            
            flDoneConfirmation = device.readline().decode()
            if self.debug: print(f"[{device.port}] FL done confirmation: {flDoneConfirmation}")
        else:
            line = ''
            while True:
                for remoteDevice in self.devices[1:]:
                    if remoteDevice.in_waiting: print(f"[{remoteDevice.port}] {remoteDevice.readline()}")
                
                if device.in_waiting:
                    line = device.readline()
                    if (b"FL_DONE" in line):
                        print(f"[SERVER] Federated learning round completed")
                        break
                    else: print(f"[{device.port}] {line.decode()[:-2]}")
        

    def answerRoutingTable(self, device: serial.Serial):
        routingTableCmd = device.read()
        if self.debug: print(f"[{device.port}] Received routing table command: {routingTableCmd}")

        nodesCount = len(self.device_address_map) - 1
        device.write(struct.pack('B', nodesCount))
        if self.debug: print(f"[{device.port}] Sent nodes count: {nodesCount}")


        for port in self.device_address_map:
            if port == device.port: continue
            
            if self.debug: print(f"[{device.port}] Sending node: {self.device_address_map[port]}")
            device.write(struct.pack('H', self.device_address_map[port])) # Address
            device.write(struct.pack('B', 1)) # Hops

    def relayModemMessage(self, device: serial.Serial, expectResponse: bool):
        targetDevice = self.readAndSendMessage(device)

        if expectResponse:
            while not targetDevice.in_waiting:
                i = 1
                if self.debug and i % 10 == 0: print("Waiting for response from receiver device...")
                time.sleep(0.01)
                i = i + 1
            self.readAndSendMessage(targetDevice, True)

    def readAndSendMessage(self, device: serial.Serial, expectingMessage: bool = False) -> serial.Serial:
        if self.debug: print(f"[{device.port}] Reading message...")

        send_message_command = device.read()
        if self.debug: print(f"[{device.port}] Send message command: {send_message_command}")
        targetAddress = struct.unpack('H', device.read(2))[0]
        if self.debug: print(f"[{device.port}] Send message targetAddress: {targetAddress}")
        messageSize = struct.unpack('H', device.read(2))[0]
        if self.debug: print(f"[{device.port}] Send message messageSize: {messageSize}")
        message = []
        for i in range(messageSize):
            byte = device.read(1)
            message.append(byte)
            device.write(byte) # echo the same value (error detection)
        device.write(struct.pack('H', messageSize)) # Confirm size

        

        # Message sending
        targetDevicePort = [port for port, address in self.device_address_map.items() if address == targetAddress][0]
        targetDevice = [device for device in self.devices if device.port == targetDevicePort][0]

        if self.debug: print(f"[{targetDevice.port}] Sending message to target: {targetDevice.port}")

        if not expectingMessage: 
            targetDevice.write(b'm')
            confirmation = targetDevice.readline().decode()
            if self.debug: print(f"[{targetDevice.port}] Reading modem message confirmation: {confirmation}")

        targetDevice.write(b'r') # Read command
        # confirmation = targetDevice.read().decode()
        # if self.debug: print(f"[{targetDevice.port}] Receive message confirmation: {confirmation}")

        targetDevice.write(struct.pack('H', self.device_address_map[device.port])) # Sender address
        targetDevice.write(struct.pack('H', messageSize)) # Message size
        # Send the message
        for i in range(messageSize):
            targetDevice.write(message[i])
        
        if self.debug: print(f"[{targetDevice.port}] Message sent!")
        
        return targetDevice

    def startExperiment(self):
        self.bitsVsPrecisionExperiment()

    def bitsVsPrecisionExperiment(self):
        # self.initDevices()

        if self.enablePlot: # Start plotting thread
            thread = threading.Thread(target=self.plot, args=["MSE Evolution"])
            thread.daemon = True
            thread.start()

        train_ini_time = time.time()
        num_batches = int(self.training_epochs/self.batchSize)

        # if self.enableTest: self.sendTestAllDevices() # Initial accuracy

        # Train the device
        for batch in range(num_batches):
            batch_ini_time = time.time()
            if self.debug: print(f"[SERVER] Sending samples batch {batch + 1}/{num_batches}")
            threads = []
            for deviceIndex, device in enumerate(self.devices):
                thread = threading.Thread(target=self.sendSamples, args=(device, deviceIndex, batch))
                thread.daemon = True
                thread.start()
                threads.append(thread)
            for thread in threads: thread.join() # Wait for all the threads to end
            if self.debug: print(f'[SERVER] Batch time: {round(time.time() - batch_ini_time, 3)}s')
            
            # time.sleep(1)

            if (batch == 0): sys.stdout.write("\033[K") # print("\r\n")
            self.doFL(self.devices[0], self.devices[1])

            if self.enableTest:
                self.sendTestAllDevices() # To calculate the accuracy on every epoch
            
            # time.sleep(2)

        if self.debug: print(f'[SERVER] Training completed in {time.time() - train_ini_time}s')

        self.plotAccuracies()
        figname = f"plots/{len(self.devices)}d-{HIDDEN_NODES}hn-{self.batchSize}bs.png"
        plt.savefig(figname, format='png')
        print(f"Generated {figname}")
