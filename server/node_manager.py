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


class NodeManager:

    def __init__(self, seed, devices):
        self.devices = devices
        self.seed = seed

        self.samples_folder = "./datasets/keywords"
        train_samples_split = 160       # Number of samples for training of each keyword
        test_samples_split = 20         # Number of samples for training of each keyword

        # Experiment sizes
        self.training_epochs = 160      # Amount of training epochs. Can't be more than kws * train_samples_split
        self.testing_epochs = 20        # Amount of test samples of each keyword. Can't be more than kws * test_samples_split

        self.momentum = 0.9
        self.learningRate= 0.05

        self.enableTest = True
        self.enablePlot = False
        self.batchSize = 20             # Must be divisble by the amount of keywords

        self.keywords_buttons = {
            "montserrat": 1,
            "pedraforca": 2,
            "vermell": 3,
            "blau": 4,
        }

        self.experiment = 'iid'        # 'iid', 'no-iid', 'train-test', None
        self.debug = False
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
                sys.exit(f"[MAIN] Not enough samples for keyword {word}")
            random.shuffle(file_list)
            files.append(list(map(lambda f: f"{word}/{f}", file_list[0:train_samples_split])))
            test_files.append(list(map(lambda f: f"{word}/{f}", file_list[train_samples_split:(train_samples_split+test_samples_split)])))

        self.keywords = list(sum(zip(*files), ()))
        self.test_keywords = list(sum(zip(*test_files), ()))

        if (self.training_epochs > len(self.keywords) / len(self.devices)):
            sys.exit(f"[MAIN] Not enough training samples for {self.training_epochs} training epochs on {len(self.devices)} devices")
        if (self.testing_epochs > len(self.test_keywords)):
            sys.exit(f"[MAIN] Not enough testing samples for {self.testing_epochs} testing epochs")
        
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
    def initializeDevices(self):
        threads = []
        for i, device in enumerate(self.devices):

            hidden_layer = np.random.uniform(-0.5,0.5, SIZE_HIDDEN_LAYER).astype('float32')
            output_layer = np.random.uniform(-0.5, 0.5, SIZE_OUTPUT_LAYER).astype('float32')

            thread = threading.Thread(target=self.initDevice, args=(hidden_layer, output_layer, device))
            thread.daemon = True
            thread.start()
            threads.append(thread)
        for thread in threads: thread.join() # Wait for all the threads to end

    def initDevice(self, hidden_layer, output_layer, device):
        device.reset_input_buffer()
        device.write(b's')
        initConfirmation = device.readline().decode()
        if self.debug: print(f"[{device.port}] Init device confirmation:", initConfirmation)
        device.write(struct.pack('i', self.seed))
        # print(f"Seed conf: {device.readline().decode()}")
        if self.debug: print(f"[{device.port}] Sending blank model...")

        device.write(struct.pack('f', self.learningRate))
        device.write(struct.pack('f', self.momentum))

        for i in trange(len(hidden_layer), desc="Sending hidden layer"):
            res = device.read() # wait until confirmation of float received
            device.write(struct.pack('f', hidden_layer[i]))
        
        for i in trange(len(output_layer), desc="Sending output"):
            res = device.read() # wait until confirmation of float received
            device.write(struct.pack('f', output_layer[i]))

        if self.debug: print(f"[{device.port}] Model sent")
        modelReceivedConfirmation = device.readline().decode()
        if self.debug: print(f"[{device.port}] Model received confirmation: {modelReceivedConfirmation}")

    # Batch size: The amount of samples to send
    def sendSamples(self, device, deviceIndex, batch_index):
        start = ((deviceIndex*self.training_epochs) + (batch_index * self.batchSize)) #%len(keywords)
        end = ((deviceIndex*self.training_epochs) + (batch_index * self.batchSize) + self.batchSize) #%len(keywords)

        if self.debug: print(f"[{device.port}] Sending samples of batch {batch_index + 1}, from {start+1} to {end}")

        for i in tqdm(range(start, end), desc=f"Sending batch {batch_index}"):
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

        if self.debug: print(f"[{device.port}] Sending {self.testing_epochs} test samples")
        for filename in tqdm(self.test_keywords[:self.testing_epochs], desc="Sending test samples"):
            if self.debug: print(f"[{device.port}] Sending test sample {self.testing_epochs}")
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

        while(True):
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
        for device_index, device in enumerate(self.devices):
            #plt.plot(training_accuracy_map[device_index], label=f"Device {device_index}")
            plt.plot(self.test_accuracies_map[device_index], label=f"Device {device_index}")

    def doFL(self):
        device = self.devices[0]
        device.write(b'>')

        fl_start_confirmation = device.readline().decode()
        if self.debug: print(f"[{device.port}] Fl start confirmation: {fl_start_confirmation}")
        nodes_count = device.readline().decode()
        # if self.debug: print(f"[{device.port}] Routing nodes count: {nodes_count}")
        if (nodes_count == "0\r\n"):
            print("No nodes found")
            exit()

        max_epochs_since_last_fl = device.readline().decode()
        # if self.debug: print(f"[{device.port}] Max epochs since last FL: {max_epochs_since_last_fl}")
        if (max_epochs_since_last_fl == "0\r\n"):
            print("No new samples on other nodes")
            exit()
        
        localWeightFactor = device.readline().decode()
        # if self.debug: print(f"[{device.port}] Local weights factor: {localWeightFactor}")
        externalWeightFactor = device.readline().decode()
        # if self.debug: print(f"[{device.port}] External weights factor: {externalWeightFactor}")
        numBatches = device.readline().decode()
        # if self.debug: print(f"[{device.port}] Num batches: {numBatches}")

        line = ''
        while True:
            if self.devices[1].in_waiting: print(f"[{self.devices[1].port}] {self.devices[1].readline()}")
            if device.in_waiting:
                line = device.readline()
                print(f"[{device.port}] {line}")
                if (b"FL_DONE" in line):
                    break
        

    def startExperiment(self):
        # self.initializeDevices()
    
        if self.enablePlot: # Start plotting thread
            thread = threading.Thread(target=self.plot, args=["MSE Evolution"])
            thread.daemon = True
            thread.start()

        train_ini_time = time.time()
        num_batches = int(self.training_epochs/self.batchSize)

        if self.enableTest: self.sendTestAllDevices() # Initial accuracy

        # Train the device
        for batch in range(num_batches):
            batch_ini_time = time.time()
            if self.debug: print(f"[MAIN] Sending batch {batch + 1}/{num_batches}")
            threads = []
            for deviceIndex, device in enumerate(self.devices):
                thread = threading.Thread(target=self.sendSamples, args=(device, deviceIndex, batch))
                thread.daemon = True
                thread.start()
                threads.append(thread)
            for thread in threads: thread.join() # Wait for all the threads to end
            if self.debug: print(f'[MAIN] Batch time: {round(time.time() - batch_ini_time, 3)}s')
            
            time.sleep(1)

            self.doFL()

            if self.enableTest:
                self.sendTestAllDevices() # To calculate the accuracy on every epoch

        if self.debug: print(f'[MAIN] Training completed in {time.time() - train_ini_time}s')

        # self.sendTestAllDevices() # Final accuracy

        self.plotAccuracies()
        figname = f"plots/{len(self.devices)}d-{HIDDEN_NODES}hn-{self.batchSize}bs.png"
        plt.savefig(figname, format='png')
        print(f"Generated {figname}")
