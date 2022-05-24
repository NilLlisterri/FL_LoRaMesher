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

seed = 4321
random.seed(seed)
np.random.seed(seed)

samples_per_device = 80 # Amount of samples of each word to send to each device
batch_size = 10 # Must be even, hsa to be split into 2 types of samples
experiment = 'iid' # 'iid', 'no-iid', 'train-test', None
use_threads = True
test_samples_amount = 40

weights_datatype = "int8" # float32
weights_unpack = "c"
weights_bytes = 1

size_hidden_nodes = 15
size_hidden_layer = (650+1)*size_hidden_nodes
size_output_layer = (size_hidden_nodes+1)*3
# hidden_layer = np.zeros(size_hidden_layer).astype('float32')
# output_layer = np.zeros(size_output_layer).astype('float32')
output_layer = np.random.uniform(-0.05, 0.05, size_output_layer).astype(weights_datatype)
hidden_layer = np.random.uniform(-0.05,0.05, size_hidden_layer).astype(weights_datatype)
momentum = 0.9      # 0 - 1
learningRate= 0.6   # 0 - 1
dropoutRate = 0     # 0 - 100
pauseListen = False # So there are no threads reading the serial input at the same time

montserrat_files = [file for file in os.listdir("datasets/mountains") if file.startswith("montserrat")]
pedraforca_files = [file for file in os.listdir("datasets/mountains") if file.startswith("pedraforca")]
vermell_files = [file for file in os.listdir("datasets/colors") if file.startswith("vermell")]
verd_files = [file for file in os.listdir("datasets/colors") if file.startswith("verd")]
blau_files = [file for file in os.listdir("datasets/colors") if file.startswith("blau")]
test_montserrat_files = [file for file in os.listdir("datasets/test/") if file.startswith("montserrat")]
test_pedraforca_files = [file for file in os.listdir("datasets/test") if file.startswith("pedraforca")]

graph = []
repaint_graph = True

random.shuffle(montserrat_files)
random.shuffle(pedraforca_files)

# mountains = []
# for index in range(0, len(montserrat_files), 5):
#     mountains.append(montserrat_files[index])
#     mountains.append(montserrat_files[index+1])
#     mountains.append(montserrat_files[index+2])
#     mountains.append(montserrat_files[index+3])
#     mountains.append(montserrat_files[index+4])

#     mountains.append(pedraforca_files[index])
#     mountains.append(pedraforca_files[index+1])
#     mountains.append(pedraforca_files[index+2])
#     mountains.append(pedraforca_files[index+3])
#     mountains.append(pedraforca_files[index+4])

mountains = list(sum(zip(montserrat_files, pedraforca_files), ()))

test_mountains = list(sum(zip(test_montserrat_files, test_pedraforca_files), ()))
random.shuffle(vermell_files)

test_vermell_files = [file for file in os.listdir("datasets/test") if file.startswith("vermell")]
test_3_classes = list(sum(zip(test_montserrat_files, test_pedraforca_files, test_vermell_files), ()))
#random.shuffle(verd_files)
#random.shuffle(blau_files)

def print_until_keyword(keyword, arduino):
    while True: 
        msg = arduino.readline().decode()
        if msg[:-2] == keyword:
            break
        # else:
            # print(f'({arduino.port}):',msg, end='')

def init_network(hidden_layer, output_layer, device, deviceIndex):
    device.reset_input_buffer()
    device.write(b's')
    print_until_keyword('start', device)
    # print(f"Sending model to {device.port}")

    device.write(struct.pack('f', learningRate))
    device.write(struct.pack('f', momentum))
    device.write(struct.pack('b', dropoutRate))
    device.write(struct.pack('f', seed))

    for i in range(len(hidden_layer)):
        #device.read() # wait until confirmation of float received
        float_num = hidden_layer[i]
        data = struct.pack('f', float_num)
        device.write(data)

    for i in range(len(output_layer)):
        #device.read() # wait until confirmation of float received
        float_num = output_layer[i]
        data = struct.pack('f', float_num)
        device.write(data)

    # print(f"Model sent to {device.port}")
    modelReceivedConfirmation = device.readline().decode()
    # print(f"Model received confirmation: ", modelReceivedConfirmation)
    
# Batch size: The amount of samples to send
def sendSamplesIID(device, deviceIndex, batch_size, batch_index, errors_queue, successes_queue):
    global montserrat_files, pedraforca_files, mountains

    start = (deviceIndex*samples_per_device) + (batch_index * batch_size)
    end = (deviceIndex*samples_per_device) + (batch_index * batch_size) + batch_size

    # print(f"[{device.port}] Sending samples from {start} to {end}")
    files = mountains[start:end]
    for i, filename in enumerate(files):
        if (filename.startswith("montserrat")):
            num_button = 1
        elif (filename.startswith("pedraforca")):
            num_button = 2
        else:
            exit("Unknown button for sample")
        # print(f"[{device.port}] Sending sample {filename} ({i}/{len(files)}): Button {num_button}")
        error, success = sendSample(device, 'datasets/mountains/'+filename, num_button, deviceIndex)
        successes_queue.put(success)
        errors_queue.put(error)
    
def sendSamplesNonIID(device, deviceIndex, batch_size, batch_index, errors_queue, successes_queue):
    global montserrat_files, pedraforca_files, vermell_files, verd_files, blau_files

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
    elif  (deviceIndex == 2):
        files = vermell_files[start:end]
        num_button = 3
        dir = 'colors'
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
        startConfirmation = device.readline().decode()
        # print(f"[{device.port}] Train start confirmation:", startConfirmation)

        device.write(struct.pack('B', num_button))
        device.readline().decode() # Button confirmation

        device.write(struct.pack('B', 1 if only_forward else 0))
        conf = device.readline().decode()
        # print(f"Only forward confirmation: {conf}") # Button confirmation

        for i, value in enumerate(data['payload']['values']):
            device.write(struct.pack('h', value))

        conf = device.readline().decode()
        print(f"[{device.port}] Sample received confirmation:", conf)

        while True: 
            input = device.readline()
            print(f"[{device.port}]", input)
        
            if input == b'LOG_END\r\n':
                break


        # while True:
        #     print(f"[{device.port}]", device.readline().decode())

        # print(f"Fordward millis received: ", device.readline().decode())
        # print(f"Backward millis received: ", device.readline().decode())
        device.readline().decode() # Accept 'graph' command
        error, num_button_predicted = read_graph(device, deviceIndex)
        #if (error > 0.28):
        #    print(f"[{device.port}] Sample {samplePath} generated an error of {error}")
        print(f'[{device.port}] Sample sent in: {(time.time()*1000)-ini_time} milliseconds)')
    return error, num_button == num_button_predicted

def sendTestSamples(device, deviceIndex, errors_queue, successes_queue, classes = 2):
    global test_mountains, test_3_classes

    start = deviceIndex*test_samples_amount
    end = (deviceIndex*test_samples_amount) + test_samples_amount
    print(f"[{device.port}] Sending test samples from {start} to {end}")

    # files = mountains[start:end]
    if classes == 2:
        files = test_mountains[start:end]
    else:
        files = test_3_classes[start:end]

    for i, filename in enumerate(files):
        if (filename.startswith("montserrat")):
            num_button = 1
        elif (filename.startswith("pedraforca")):
            num_button = 2
        elif (filename.startswith("vermell")):
            num_button = 3
        else:
            exit("Unknown button for sample")
        # print(f"[{device.port}] Sending sample {filename} ({i}/{len(files)}): Button {num_button}")
        # error, success = sendSample(device, 'datasets/mountains/'+filename, num_button, deviceIndex, True)
        error, success = sendSample(device, 'datasets/test/'+filename, num_button, deviceIndex, True)
        errors_queue.put(error)
        successes_queue.put(success)

def sendTestAllDevices(classes = 2):
    global devices
    errors_queue = Queue()
    successes_queue = Queue()
    for deviceIndex, device in enumerate(devices):
        if use_threads:
            thread = threading.Thread(target=sendTestSamples, args=(device, deviceIndex, errors_queue, successes_queue, classes))
            thread.daemon = True
            thread.start()
            threads.append(thread)
        else:
            sendTestSamples(device, deviceIndex, errors_queue, successes_queue, classes)
    for thread in threads: 
        thread.join() # Wait for all the threads to end

    test_accuracy = sum(successes_queue.queue)/len(successes_queue.queue)
    # print(f"Testing loss mean: {sum(errors_queue.queue)/len(errors_queue.queue)}")
    print(f"Testing accuracy: {test_accuracy}")
    print(f"{test_accuracy}, ", end = '')

def read_graph(device, deviceIndex):
    global repaint_graph

    outputs = device.readline().decode().split()
    error = device.readline().decode()

    print(f"Outputs: ", outputs)
    print(f"Error received: ", error)

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
        plt.ylabel('Loss') # or Error
        plt.xlabel('Epoch')
        # plt.axes().set_ylim([0, 0.6])
        # plt.xlim(bottom=0)
        # plt.autoscale()

        if (experiment == 'train-test'):
            plt.axvline(x=samples_per_device)

        repaint_graph = False

    plt.pause(2)

def listenDevice(device, deviceIndex):
    global pauseListen, graph
    while True:
        while (pauseListen):
            print("Paused...")
            time.sleep(0.1)

        d.timeout = None
        msg = device.readline().decode()
        if (len(msg) > 0):
            print(f'({device.port}):', msg, end="")
            # Modified to graph
            if msg[:-2] == 'graph':
                read_graph(device, deviceIndex)

            elif msg[:-2] == 'start_fl':
                startFL()

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

def FlGetModel(d, device_index, devices_hidden_layer, devices_output_layer, devices_num_epochs, old_devices_connected):
    global size_hidden_layer, size_output_layer
    d.reset_input_buffer()
    d.reset_output_buffer()
    d.timeout = 5

    print(f'Starting connection to {d.port} ...') # Hanshake
    d.write(b'>') # Python --> SYN --> Arduino
    if d.read() == b'<': # Python <-- SYN ACK <-- Arduino
        d.write(b's') # Python --> ACK --> Arduino
        
        print('Connection accepted.')
        devices_connected.append(d)
        #devices_hidden_layer = np.vstack((devices_hidden_layer, np.empty(size_hidden_layer)))
        #devices_output_layer = np.vstack((devices_output_layer, np.empty(size_output_layer)))
        d.timeout = None

        print_until_keyword('start', d)
        devices_num_epochs.append(int(d.readline()[:-2]))

        print(f'Receiving model from {d.port} ...')
        ini_time = time.time()

        for i in range(size_hidden_layer): # hidden layer
            data = d.read(weights_bytes)
            num = int.from_bytes(data, "big", signed=True);
            #[num] = struct.unpack(weights_unpack, data)
            devices_hidden_layer[device_index][i] = num

        for i in range(size_output_layer): # output layer
            data = d.read(weights_bytes)
            num = int.from_bytes(data, "big", signed=True);
            #[num] = struct.unpack(weights_unpack, data)
            devices_output_layer[device_index][i] = num

        print(f'Model received from {d.port} ({time.time()-ini_time} seconds)')

        # if it was not connected before, we dont use the devices' model
        if not d in old_devices_connected:
            devices_num_epochs[device_index] = 0
            print(f'Model not used. The device {d.port} has an outdated model')

    else:
        print(f'Connection timed out. Skipping {d.port}.')

def sendModel(d, hidden_layer, output_layer):
    print("Sending model")
    ini_time = time.time()
    for i in range(size_hidden_layer): # hidden layer
        #d.read() # wait until confirmatio
        num = hidden_layer[i]
        #data = struct.pack('f', num)
        data = int(num).to_bytes(1, "big", signed=True)
        d.write(data)

    for i in range(size_output_layer): # output layer
        #d.read() # wait until confirmatio
        num = output_layer[i]
        #data = struct.pack('f', num)
        data = int(num).to_bytes(1, "big", signed=True)
        d.write(data)

    print(f'Model sent to {d.port} ({time.time()-ini_time} seconds)')

def startFL():
    global devices_connected, hidden_layer, output_layer, pauseListen

    pauseListen = True

    # print('Starting Federated Learning')
    old_devices_connected = devices_connected
    devices_connected = []
    devices_hidden_layer = np.empty((len(devices), size_hidden_layer), dtype=weights_datatype)
    devices_output_layer = np.empty((len(devices), size_output_layer), dtype=weights_datatype)
    devices_num_epochs = []

    ##################
    # Receiving models
    ##################
    threads = []
    for i, d in enumerate(devices):
        if use_threads:
            thread = threading.Thread(target=FlGetModel, args=(d, i, devices_hidden_layer, devices_output_layer, devices_num_epochs, old_devices_connected))
            thread.daemon = True
            thread.start()
            threads.append(thread)
        else:
            FlGetModel(d, i, devices_hidden_layer, devices_output_layer, devices_num_epochs, old_devices_connected)
 
    for thread in threads: thread.join() # Wait for all the threads to end

    
    ####################
    # Processing models
    ####################

    # if sum == 0, any device made any epoch
    if sum(devices_num_epochs) > 0:
        # We can use weights to change the importance of each device
        # example weights = [1, 0.5] -> giving more importance to the first device...
        # is like percentage of importance :  sum(a * weights) / sum(weights)
        # ini_time = time.time() * 1000
        hidden_layer = np.average(devices_hidden_layer, axis=0, weights=devices_num_epochs)
        output_layer = np.average(devices_output_layer, axis=0, weights=devices_num_epochs)
        # print(f'Average millis: {(time.time()*1000)-ini_time} milliseconds)')

    print("Layers averaged")

    #################
    # Sending models
    #################
    threads = []
    for d in devices_connected:
        # print(f'Sending model to {d.port} ...')

        if use_threads:
            thread = threading.Thread(target=sendModel, args=(d, hidden_layer, output_layer))
            thread.daemon = True
            thread.start()
            threads.append(thread)
        else:
            sendModel(d, hidden_layer, output_layer)
    for thread in threads: thread.join() # Wait for all the threads to end
    pauseListen = False

getDevices()

# Send the blank model to all the devices
# threads = []
# for i, d in enumerate(devices):
#     if use_threads:
#         thread = threading.Thread(target=init_network, args=(hidden_layer, output_layer, d, i))
#         thread.daemon = True
#         thread.start()
#         threads.append(thread)
#     else:
#         init_network(hidden_layer, output_layer, d, i)
# for thread in threads: thread.join() # Wait for all the threads to end

# print(f"Initial model sent to all devices")

if experiment != None:
    # Train the device
    threads = []
    train_ini_time = time.time()
    errors_queue = Queue() # MSE errors
    successes_queue = Queue() # Amount of right inferences
    batches = int(samples_per_device/batch_size)
    for batch in range(batches):
        # batch_ini_time = time.time()
        for deviceIndex, device in enumerate(devices):
            if experiment == 'iid' or experiment == 'train-test':
                method = sendSamplesIID            
            elif experiment == 'no-iid':
                method = sendSamplesNonIID

            if use_threads:
                thread = threading.Thread(target=method, args=(device, deviceIndex, batch_size, batch, errors_queue, successes_queue))
                thread.daemon = True
                thread.start()
                threads.append(thread)
            else:
                method(device, deviceIndex, batch_size, batch, errors_queue, successes_queue)
        for thread in threads: thread.join() # Wait for all the threads to end

        # print(f'Batch time: {time.time() - batch_ini_time} seconds)')

        #sendTestAllDevices(3)

        fl_ini_time = time.time()
        startFL()
        print(f'FL time: {time.time() - fl_ini_time} seconds)')

    #print(f"Training error: {sum(errors_queue.queue)/len(errors_queue.queue)}")

    print(f"Training successes: {sum(successes_queue.queue)/len(successes_queue.queue)}")

    train_time = time.time()-train_ini_time
    # print(f'Trained in ({train_time} seconds)')

    if experiment == 'train-test':
        sendTestAllDevices(2)

    for device in devices:
        device.close()

# Listen their updates
#for i, d in enumerate(devices):
#    thread = threading.Thread(target=listenDevice, args=(d, i))
#    thread.daemon = True
#    thread.start()


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



