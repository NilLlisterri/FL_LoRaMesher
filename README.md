# FL LoRaMesher

<p align="center">
Federated Learning via LoRaMesher with a two-board application/modem design.
</p>

This repository contains the code for the Arduino Portenta H7 application microcontroller and the code for the Python server that will orchestrate the experiment.
The code for the modem microcontroller can be found in [here](https://github.com/NilLlisterri/TTGO-LoRaMesher/tree/master).

## Structure

The microcontroller code is in the `src` folder, and the server is in `server`.
The `lib` folder contains the EdgeImpulse SDK, basically used to extract the MFCC from the audio samples.
The pre-recorded audio samples are stored in the `datasets/keywords` folder.

## Experiments
The experiments are configured in the `node_manager.py` file. 
Constants, as the size of the NN, are found in the `server/constants.py` file. When the size of the NN is changed there, it also has to be updated in the `src/config.h`.

In the experiments, samples can be sent to all of the nodes in parallel, FL rounds can be triggered on specific nodes and metrics and plots can be generated.

### Important methods

* `server/node_manager.py`
    * `startExperiment`: The main method that will orchestrate the experiment.
    * `sendSamples`: Send a specific amount of training samples to the nodes.
    * `sendTestAllDevices`: Send the configured amount of test samples to all devices to obtain the accuracy of the model.
    * `doFL`: Trigger a FL round in a node. The node will query all the other nodes and perform FL with the best candidate.

* `src/main.cpp`
    * `sendModemMessage`: Send a message to another node, through the modem connected via serial.
    * `getModemMessage`: Expect a response to a message from another node, for example to verify the successfull delivery of a previous message.
    * `doFL`: Start the FL process.
    * `getRoutingTable`: Obtain the list of reachable nodes from the modem.
    * `requestWeights`/`sendWeights`: Send a modem message to obtain the weights from another node to update the local weights or viceversa.
    * `sendMetrics`: Send the node metrics to a node so it can choose whether to perform FL or not.


## Requirements
* Visual Studio Code
* PlaformIO extension
* Python 3

## Execution
To configure the experiments, first update the `devices` variable in the `server/experiment.py` file with the ports of the devices you are going to use.

To build and flash the code to the Arduino Portenta H7, execute this command updating the port for each board you are going to use.
```sh
pio run --target upload -e portenta_h7_m7 --upload-port PORT
```

Finally, to execute the experiments, execute the command
```sh
python .\server\experiment.py`
```
(or simply `.\server\experiment.py` if you assigned execution permission to the file).
