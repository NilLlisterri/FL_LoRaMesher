# FL LoRaMesher

<p align="center">
Federated Learning via LoRaMesher with a two-board application/modem design.
</p>

This repository contains the code for the Arduino Portenta H7 application microcontroller and the Python server that orchestrates the experiments. The nodes train a keyword spotting (KWS) neural network on-device and share their model weights through Federated Learning rounds.

The code for the modem microcontroller (TTGO LoRa32 or TTGO T-Beam, both running LoRaMesher) can be found [here](https://github.com/NilLlisterri/TTGO-LoRaMesher/tree/master).

## Overview

Each node runs a small feed-forward neural network (650 input nodes from MFCC features, 20 hidden neurons, 4 output nodes) that classifies four keywords: *montserrat*, *pedraforca*, *vermell* and *blau*. The audio samples are pre-recorded and streamed to the boards over serial by the server; the EdgeImpulse SDK extracts the MFCC features on-device.

During an FL round, a node queries the other reachable nodes for their metrics (training epochs since the last FL round), picks the best candidate (or a specific target), requests its weights and merges them into the local model with a weighted average proportional to the number of new samples each model has seen.

### Weight quantization

To reduce the payload size over LoRa, weights are quantized before transmission. The number of quantization bits is configurable per FL round from the server (e.g. 12 bits instead of 32-bit floats). Weights are transferred in batches (200 weights per batch, the LoRaMesher payload limit); each batch is dynamically scaled using its own min/max weight values, bit-packed, and de-scaled on the receiving end.

### Serial modem passthrough

For experimentation and debugging, the boards can run without LoRa hardware: with `useSerialModemPassthrough` enabled (in `src/main.cpp` and `node_manager.py`), the server relays the modem messages between the boards over their serial connections, emulating the mesh network. Set it to `false` on both sides to use the real LoRaMesher modem over `Serial1`.

## Structure

* `src`: The Portenta H7 application code (`main.cpp`, the neural network in `NN.h`, and `config.h`).
* `server`: The Python server. `experiment.py` is the entry point, `node_manager.py` orchestrates the experiments, and `constants.py` holds shared constants.
* `lib`: The EdgeImpulse SDK, used to extract the MFCC features from the audio samples.
* `datasets/keywords`: The pre-recorded audio samples, one folder per keyword.

Constants such as the NN size are defined in `server/constants.py`. When changed there, they must also be updated in `src/config.h` and `src/NN.h`.

## Experiments

Experiments are configured in `node_manager.py` (training/testing epochs, batch size, learning rate, momentum, IID/non-IID split, plotting). The current experiment (`bitsVsPrecisionExperiment`) measures the effect of the quantization bit-depth on model accuracy: it interleaves training batches with FL rounds and test evaluations, and saves an accuracy plot to the `plots` folder.

### Important methods

* `server/node_manager.py`
    * `startExperiment`: Entry point that runs the configured experiment.
    * `initDevices`: Send the same randomly-initialized model to all devices.
    * `sendSamples` / `sendTestAllDevices`: Send training samples to a node, or test samples to all nodes to measure accuracy.
    * `doFL`: Trigger an FL round on a node, optionally specifying the target device and the quantization bits.
    * `relayModemMessage` / `answerRoutingTable`: Emulate the modem when the serial passthrough mode is enabled.
* `src/main.cpp`
    * `sendModemMessage` / `getModemMessage`: Send a message to another node through the modem (with byte-level echo verification), or receive one.
    * `doFL`: Run the FL process: get the routing table, collect metrics, pick the best candidate and merge its weights.
    * `getRoutingTable`: Obtain the list of reachable nodes from the modem.
    * `requestWeights` / `sendWeights`: Request the quantized weight batches from another node, or scale, bit-pack and send the local ones.
    * `sendMetrics`: Report the local epoch count so the requesting node can pick its FL partner.

### Serial commands

The boards accept single-character commands over serial: `i` (initialize model with server-provided weights), `t` (train with a sample), `>` (start an FL round), `r` (print the routing table), `z` (dump the weights), `x` (print the epoch count) and `m` (a modem message is available, passthrough mode only).

## Requirements

* Visual Studio Code with the PlatformIO extension
* Python 3, with `pyserial`, `numpy`, `matplotlib`, `tqdm`, `scipy` and `python-dotenv`

## Execution

To configure the experiments, first update the `devices` list and the `device_address_map` in `server/experiment.py` with the serial ports of the boards you are going to use.

To build and flash the code to each Arduino Portenta H7, run this command updating the port for each board:
```sh
pio run --target upload -e portenta_h7_m7 --upload-port PORT
```

Finally, run the experiment:
```sh
python server/experiment.py
```
