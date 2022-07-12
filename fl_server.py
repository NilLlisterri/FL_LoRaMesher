from sys import float_info
import serial
from serial.tools.list_ports import comports

import struct
import time
import numpy as np
import matplotlib.pyplot as plt
import threading
import time
import json
import os
import random
from queue import Queue
import time

debug = True

seed = 4321
random.seed(seed)
np.random.seed(seed)

samples_per_device = 120 # Amount of samples of each word to send to each device
batch_size = 60 # Must be even, hsa to be split into 2 types of samples
experiment = 'train-test' # 'iid', 'no-iid', 'train-test', None
use_threads = True
test_samples_amount = 40

# Options: int8, c, 1 / float32 f, 4 / int16, h, 2
weights_datatype = "int8"
weights_unpack = "c"
weights_bytes = 1

size_hidden_nodes = 15

momentum = 0.9      # 0 - 1
learningRate= 0.6   # 0 - 1
dropoutRate = 0     # 0 - 100
pauseListen = False # So there are no threads reading the serial input at the same time

montserrat_files = [file for file in os.listdir("datasets/mountains") if file.startswith("montserrat")]
pedraforca_files = [file for file in os.listdir("datasets/mountains") if file.startswith("pedraforca")]
test_montserrat_files = [file for file in os.listdir("datasets/test/") if file.startswith("montserrat")]
test_pedraforca_files = [file for file in os.listdir("datasets/test") if file.startswith("pedraforca")]

graph = []
repaint_graph = True

random.shuffle(montserrat_files)
random.shuffle(pedraforca_files)

mountains = list(sum(zip(montserrat_files, pedraforca_files), ()))
test_mountains = list(sum(zip(test_montserrat_files, test_pedraforca_files), ()))


def print_until_keyword(keyword, arduino):
    while True: 
        msg = arduino.readline().decode()
        if msg[:-2] == keyword:
            break
        # else:
            # print(f'({arduino.port}):',msg, end='')
    
# Batch size: The amount of samples to send
def sendSamplesIID(device, deviceIndex, batch_size, batch_index, errors_queue, successes_queue):
    global montserrat_files, pedraforca_files, mountains

    start =  (deviceIndex*samples_per_device) + (batch_index * batch_size)
    end = (deviceIndex*samples_per_device) + (batch_index * batch_size) + batch_size

    if (debug): print(f"[{device.port}] Sending samples from {start} to {end}")

    files = mountains[start:end]
    for i, filename in enumerate(files):
        if (filename.startswith("montserrat")):
            num_button = 1
        elif (filename.startswith("pedraforca")):
            num_button = 2
        else:
            exit("Unknown button for sample")
        
        if (debug): print(f"[{device.port}] Sending sample {filename} ({i}/{len(files)}): Button {num_button}")
        error, success = sendSample(device, 'datasets/mountains/'+filename, num_button, deviceIndex)
        successes_queue.put(success)
        errors_queue.put(error)
    
def sendSamplesNonIID(device, deviceIndex, batch_size, batch_index, errors_queue, successes_queue):
    global montserrat_files, pedraforca_files

    start = (batch_index * batch_size)
    end = (batch_index * batch_size) + batch_size

    if (deviceIndex == 0):
        files = montserrat_files[start:end]
        num_button = 1
        dir = 'mountains'
    elif  (deviceIndex == 1):
        files = pedraforca_files[start:end]
        num_button = 2
        dir = 'mountains'
    else:
        exit("Exceeded device index")

    for i, filename in enumerate(files):
        print(f"[{device.port}] Sending sample {filename} ({i}/{len(files)}): Button {num_button}")
        error, success = sendSample(device, f"datasets/{dir}/{filename}", num_button, deviceIndex)
        successes_queue.put(success)
        errors_queue.put(error)

def sendSample(device, samplePath, num_button, deviceIndex, only_forward = False):
    with open(samplePath) as f:
        ini_time = time.time() * 1000
        data = json.load(f)
        device.write(b't')
        #startConfirmation = device.readline().decode()
        #if (debug): print(f"[{device.port}] Train start confirmation: {startConfirmation}")
        

        device.write(struct.pack('B', num_button))
        butConf = device.readline().decode() # Button confirmation
        if (debug): print(f"[{device.port}] Button confirmation: {butConf}")
        
        device.write(struct.pack('B', 1 if only_forward else 0))
        conf = device.readline().decode()
        if (debug): print(f"[{device.port}] Only forward confirmation: {conf}") # Button confirmation

        for i, value in enumerate(data['payload']['values']):
            device.write(struct.pack('h', value))

        conf = device.readline().decode()
        print(f"[{device.port}] Sample received confirmation:", conf)

        while True: 
            input = device.readline()
            print(f"[{device.port}]", input)
        
            if input == b'LOG_END\r\n':
                break

        device.readline().decode() # Accept 'graph' command
        error, num_button_predicted = read_graph(device, deviceIndex)
        print(f'[{device.port}] Sample sent in: {(time.time()*1000)-ini_time} milliseconds)')
    return error, num_button == num_button_predicted

def sendTestSamples(device, deviceIndex):
    global test_mountains

    errors_queue = Queue()
    successes_queue = Queue()

    start = deviceIndex*test_samples_amount
    end = (deviceIndex*test_samples_amount) + test_samples_amount
    print(f"[{device.port}] Sending test samples from {start} to {end}")

    
    files = test_mountains[start:end]

    for i, filename in enumerate(files):
        if (filename.startswith("montserrat")):
            num_button = 1
        elif (filename.startswith("pedraforca")):
            num_button = 2
        else:
            exit("Unknown button for sample")
        # print(f"[{device.port}] Sending sample {filename} ({i}/{len(files)}): Button {num_button}")
        # error, success = sendSample(device, 'datasets/mountains/'+filename, num_button, deviceIndex, True)
        error, success = sendSample(device, 'datasets/test/'+filename, num_button, deviceIndex, True)
        errors_queue.put(error)
        successes_queue.put(success)
    
    print(f"[{device.port}] Testing loss total: {sum(errors_queue.queue)}")
    print(f"[{device.port}] Testing loss mean: {sum(errors_queue.queue)/len(errors_queue.queue)}")
    print(f"[{device.port}] Testing accuracy: {sum(successes_queue.queue)/len(successes_queue.queue)}")

def sendTestAllDevices():
    global devices
    for deviceIndex, device in enumerate(devices):
        if use_threads:
            thread = threading.Thread(target=sendTestSamples, args=(device, deviceIndex))
            thread.daemon = True
            thread.start()
            threads.append(thread)
        else:
            sendTestSamples(device, deviceIndex)
    for thread in threads: 
        thread.join() # Wait for all the threads to end

    

def read_graph(device, deviceIndex):
    global repaint_graph

    outputs = device.readline().decode().split()
    error = device.readline().decode()

    # print(f"[{device.port}] Outputs: ", outputs)
    # print(f"[{device.port}] Error received: ", error)

    ne = device.readline()[:-2]
    n_epooch = int(ne)

    n_error = device.read(4)
    [n_error] = struct.unpack('f', n_error)
    nb = device.readline()[:-2]
    graph.append([n_epooch, n_error, deviceIndex])
    repaint_graph = True
    return n_error, outputs.index(max(outputs)) + 1

def read_number(msg):
    while True:
        try:
            #return 2;
            return int(input(msg))
        except:
            print("ERROR: Not a number")

def read_port(msg):
    while True:
        try:
            port = input(msg)
            #port = "COM3";
            return serial.Serial(port, 115200)
        except:
            print(f"ERROR: Wrong port connection ({port})")

def plot_graph():
    global graph, repaint_graph, devices

    if (repaint_graph):
        colors = ['r', 'g', 'b', 'y']
        markers = ['-', '--', ':', '-.']
        #devices =  [x[2] for x in graph]        
        for device_index, device in enumerate(devices):
            epoch = [x[0] for x in graph if x[2] == device_index]
            error = [x[1] for x in graph if x[2] == device_index]
        
            plt.plot(error, colors[device_index] + markers[device_index], label=f"Device {device_index}")

        plt.legend()
        plt.xlim(left=0)
        plt.ylim(bottom=0, top=0.7)
        plt.ylabel('') # or Error
        plt.xlabel('')
        # plt.axes().set_ylim([0, 0.6])
        # plt.xlim(bottom=0)
        # plt.autoscale()

        # if (experiment == 'train-test'):
        #     plt.axvline(x=samples_per_device, color='blue')
        
        plt.axvline(x=30, color='orange')
        plt.axvline(x=60, color='orange')
        plt.axvline(x=90, color='orange')
        plt.axvline(x=120, color='b')

        # batches = int(samples_per_device/batch_size)
        # for batch in range(batches):
        #     pos = batch*samples_per_device
        #     plt.axvline(x=pos, color='orange')

        repaint_graph = False

    plt.pause(2)

def listenDevice(device, deviceIndex):
    global pauseListen, graph
    while True:
        while (pauseListen):
            print("Paused...")
            time.sleep(0.1)

        msg = device.readline().decode()
        if (len(msg) > 0):
            print(f'({device.port}):', msg, end="")
            # Modified to graph
            if msg[:-2] == 'graph':
                read_graph(device, deviceIndex)


def getDevices():
    global devices, devices_connected
    num_devices = read_number("Number of devices: ")

    available_ports = comports()
    print("Available ports:")
    for available_port in available_ports:
        print(available_port)

    devices = [read_port(f"Port device_{i+1}: ") for i in range(num_devices)]
    print("Ports opened successfully")
    devices_connected = devices


getDevices()

if experiment != None:
    # Train the device
    threads = []
    train_ini_time = time.time()
    errors_queue_map = {}
    successes_queue_map = {}
    for deviceIndex, device in enumerate(devices):
        errors_queue_map[deviceIndex] = Queue() # MSE errors
        successes_queue_map[deviceIndex] = Queue() # Amount of right inferences

    batches = int(samples_per_device/batch_size)
    for batch in range(batches):
        if (debug): print(f"Sending batch {batch}")
        batch_ini_time = time.time()
        for deviceIndex, device in enumerate(devices):
            if experiment == 'iid' or experiment == 'train-test':
                method = sendSamplesIID            
            elif experiment == 'no-iid':
                method = sendSamplesNonIID

            thread = threading.Thread(target=method, args=(device, deviceIndex, batch_size, batch, errors_queue_map[deviceIndex], successes_queue_map[deviceIndex]))
            thread.daemon = True
            thread.start()
            threads.append(thread)
            

        for thread in threads: thread.join() # Wait for all the threads to end

        print(f'Batch time: {time.time() - batch_ini_time} seconds)')

        fl_index = 0
        # def read_until_fl_end():
        #     while True: 
        #         for device_index, device in enumerate(devices):
        #             while device.in_waiting:
        #                 line = device.readline()
        #                 print(f"[{device.port}]", line, b'FL Done' in line)
        #                 if b'FL Done' in line:
        #                     return

        # #if batch < batches - 1:
        # # for device_index, device in enumerate(devices): print(f"{device_index}: {device.port}")
        # # fl_index = input("FL device index:")
        
        # fl_times = len(devices)-1
        # # FL with all devices
        # for i in range(fl_times):
        #     print(f"Starting FL number {i+1}")
        #     devices[fl_index].write(b'>')
        #     read_until_fl_end()


        
        for device_index, device in enumerate(devices):
            if device_index == fl_index: continue

            devices[fl_index].write(b'>')
            print(f"[S] Started FL process")
            
            #t_end = time.time() + 15
            #while time.time() < t_end:
            #    if devices[fl_index].in_waiting: print(devices[fl_index].readline())
            while True:
               line = devices[fl_index].readline()
               print(line)
               if line == b'READY\r\n': break

            print(f"[S] Requesting weights from {device.port}")
            device.write(b'z')

            for i in range(((650+1)*size_hidden_nodes) + (size_hidden_nodes+1)*3):
                # print(f"[S] Reading byte {i} and sending it")
                weight = device.read()
                # print(f"Value: {weight}")
                devices[fl_index].write(weight)
                conf = devices[fl_index].readline()
                if i % 1000 == 0: print(f"[S] Recevied confirmation: {conf}")

            print(f"Fl done confirmation: {devices[fl_index].readline()}")

    for deviceIndex, device in enumerate(devices):
        print(f"[{device.port}] Training loss total: {sum(errors_queue_map[deviceIndex].queue)}")
        print(f"[{device.port}] Training loss mean: {sum(errors_queue_map[deviceIndex].queue)/len(errors_queue_map[deviceIndex].queue)}")
        print(f"[{device.port}] Training accuracy: {sum(successes_queue_map[deviceIndex].queue)/len(successes_queue_map[deviceIndex].queue)}")

    train_time = time.time()-train_ini_time
    print(f'Trained in ({train_time} seconds)')

    if experiment == 'train-test':
        sendTestAllDevices()

    for device in devices:
        device.close()

# Listen their updates
#for i, d in enumerate(devices):
#    thread = threading.Thread(target=listenDevice, args=(d, i))
#    thread.daemon = True
#    thread.start()

plt.figure(figsize=(8, 4))
plt.ion()
# plt.title(f"Loss vs Epoch")
plt.show()

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

plot_graph()

if experiment != None:
    figname = f"plots/BS{batch_size}-LR{learningRate}-M{momentum}-HL{size_hidden_nodes}-DR-{dropoutRate}-S{seed}-TT{train_time}-{experiment}.png"
    plt.savefig(figname, format='png')
    print(f"Generated {figname}")

    exit()

while True:
    #if (repaint_graph): 
    plot_graph()
        #repaint_graph = False
    # time.sleep(0.1)



