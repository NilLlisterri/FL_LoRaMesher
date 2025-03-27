#include "config.h"
#include "Arduino.h"

#define EIDSP_QUANTIZE_FILTERBANK   0
#include <training_kws_inference.h>

#include <mbed.h>
#include "NN.h"
#include <map>
#include <vector>

bool debugEnabled = false;
bool useSerialModemPassthrough = true; // Experimentation and debugging
HardwareSerial& modem = useSerialModemPassthrough ? Serial : Serial1;
typedef uint16_t scaledType;

struct Metric {
  uint8_t hops;
  uint16_t num_epochs;
};

struct Node {
  uint16_t address;
  uint8_t hops;
};

/* 
  This variable will hold the recorded audio.
  Ideally this would only hold the features extracted in this core, but we have to support serial training on the M7, and that would require that bot cores knew how to extract the features
  Datasheet: https://www.st.com/resource/en/reference_manual/dm00176879-stm32h745-755-and-stm32h747-757-advanced-arm-based-32-bit-mcus-stmicroelectronics.pdf
  In this region (SRAM3) we can use 32767 bytes (32kB)
  "AHB SRAM3 is mapped at address 0x3004 0000 and accessible by all system
  masters except BDMA through D2 domain AHB matrix. AHB SRAM3 can be used
  as buffers to store peripheral input/output data for Ethernet and USB, or as shared
  memory between the two cores."
*/
struct sharedMemory {
  int16_t audio_input_buffer[16000];
};
volatile struct sharedMemory * const shared_ptr = (struct sharedMemory *)0x30040000;
NeuralNetwork<NN_HIDDEN_NEURONS>* network = new NeuralNetwork<NN_HIDDEN_NEURONS>();
uint16_t num_epochs = 0;

void getScaleRange(float &a, float &b, uint8_t scaled_weights_bits) {
    a = 0;
    b = std::pow(2, scaled_weights_bits)-1;
} 

scaledType scaleWeight(float min_w, float max_w, float weight, uint8_t scaled_weights_bits) {
    float a, b;
    getScaleRange(a, b, scaled_weights_bits);
    return round(a + ( (weight-min_w)*(b-a) / (max_w-min_w) ));
}

float deScaleWeight(float min_w, float max_w, scaledType weight, uint8_t scaled_weights_bits) {
    float a, b;
    getScaleRange(a, b, scaled_weights_bits);
    return min_w + ( (weight-a)*(max_w-min_w) / (b-a) );
}

std::vector<float> getRawWeights(uint16_t batchNum, float &min_w, float &max_w) {
  std::vector<float> weights;

  float* hidden_weights = network->getHiddenWeights();
  float* output_weights = network->getOutputWeights();

  uint start = batchNum * WEIGHTS_BATCH_SIZE;
  for(uint i = start; i < start + WEIGHTS_BATCH_SIZE; i++) {
    float weight;
    if (i > network->getHiddenWeightsAmt() + network->getOutputWeightsAmt()) break;
    if (i < network->getHiddenWeightsAmt()) {
      weight = hidden_weights[i];
    } else {
      weight = output_weights[i-network->getHiddenWeightsAmt()];
    }
    weights.push_back(weight);

    if (i == start || min_w > weight) min_w = weight;
    if (i == start || max_w < weight) max_w = weight;
  }
  return weights;
}

std::vector<scaledType> getScaledWeights(uint16_t batchNum, float &min_w, float &max_w, uint8_t scaled_weights_bits) {
  if (debugEnabled) Serial.println("Getting weights values for batch " + String(batchNum));
  std::vector<float> raw_weights = getRawWeights(batchNum, min_w, max_w);

  std::vector<scaledType> weights;
  for(uint i = 0; i < raw_weights.size(); i++) {
    scaledType scaledWeight = scaleWeight(min_w, max_w, raw_weights[i], scaled_weights_bits);
    weights.push_back(scaledWeight);
  }

  if (debugEnabled) Serial.println("Got the weights for batch " + String(batchNum) + ". Weights: " + String(weights.size()));
  return weights;
}

// I don't like this cast to non-volatile...
static int get_input_data(size_t offset, size_t length, float *out_ptr) {
  numpy::int16_to_float((const int16_t*)&shared_ptr->audio_input_buffer[offset], out_ptr, length);
  return 0;
}

void train(int nb, bool only_forward) {
  signal_t signal;
  signal.total_length = EI_CLASSIFIER_RAW_SAMPLE_COUNT;
  signal.get_data = &get_input_data;
  ei::matrix_t features_matrix(1, EI_CLASSIFIER_NN_INPUT_FRAME_SIZE);

  EI_IMPULSE_ERROR r = get_one_second_features(&signal, &features_matrix, false);
  if (r != EI_IMPULSE_OK) {
    Serial.println("ERR: Failed to get features (" + String(r) + ")");
    return;
  }

  float myTarget[network->OutputNodes] = {0};
  myTarget[nb - 1] = 1.f; // button 1 -> {1,0,0,0};  button 2 -> {0,1,0,0}, ...

  float forward_error = network->forward(features_matrix.buffer, myTarget);

  if (!only_forward) {
    network->backward(features_matrix.buffer, myTarget);
    num_epochs++;
  }

  Serial.println("graph");
  for (size_t i = 0; i < network->OutputNodes; i++) {
    Serial.print(network->get_output()[i], 5);
    Serial.print(" ");
  }
  Serial.print("\n");
  Serial.println(forward_error, 5);
  Serial.println(num_epochs, DEC);
  Serial.println(nb, DEC);
}

void trainWithSerialSample() {
  while (!Serial.available()) {}
  uint8_t num_button = Serial.read();
  Serial.print("Button "); Serial.println(num_button);

  while (!Serial.available()) {}
  bool only_forward = Serial.read() == 1;
  Serial.print("Only forward "); Serial.println(only_forward);

  byte ref[2];
  for (int i = 0; i < EI_CLASSIFIER_RAW_SAMPLE_COUNT; i++) {
    while (Serial.available() < 2) {}
    Serial.readBytes(ref, 2);
    shared_ptr->audio_input_buffer[i] = 0;
    shared_ptr->audio_input_buffer[i] = (ref[1] << 8) | ref[0];
  }
  Serial.print("Sample received for button ");
  Serial.println(num_button);

  train(num_button, only_forward);
}

volatile bool lock_modem = false;

void waitForModemBytes(uint count, String waitingMessage) {
  int i = 10;
  while (modem.available() < count) {
      if (debugEnabled && i % 10 == 0) Serial.println(waitingMessage);
      delay(5);
      i++;
    }
}

std::vector<Node> getRoutingTable() {
  lock_modem = true;
  std::vector<Node> nodes;
  if (debugEnabled) Serial.println("Getting routing table");
  modem.write('r'); // Send command
  waitForModemBytes(1, "Waiting for nodes count");
  uint8_t count = modem.read();
  if (debugEnabled) Serial.println("Number of nodes: " + String(count));
  for (int i = 0; i < count; i++) {
    waitForModemBytes(2, "Waiting for node address");
    uint16_t address;
    modem.readBytes((char*)&address, 2);
    uint8_t hops;
    modem.readBytes((char*)&hops, 1);
    Node node;
    node.address = address;
    node.hops = hops;
    nodes.push_back(node);
    if (debugEnabled) Serial.println("Node " + String(i + 1) + ": " + String(nodes[i].address));
  }
  lock_modem = false;
  return nodes;
}

int getModemMessage(byte*& bytesPtr, uint16_t &sender) {
  uint limit = millis() + 10000;
  while (millis() < limit) {
    if (modem.available()) {
      char command = modem.read();
      if (command != 'r') {
        Serial.println("Unknown command received: " + String(command));
        while(true) {
          if (Serial.available()) Serial.println(modem.read());
        }
      }

      // Read from
      waitForModemBytes(2, "Waiting for sender");
      modem.readBytes((byte*) &sender, 2);
      if (debugEnabled) Serial.println("Sender address: " + String(sender));
      
      // Read message bytes count
      waitForModemBytes(2, "Waiting for bytes count");
      uint16_t bytesCount = 0;
      modem.readBytes((uint8_t*) &bytesCount, 2);
      if (debugEnabled) Serial.println("Bytes count: " + String(bytesCount));

      bytesPtr = (byte*) malloc(bytesCount);
      if (debugEnabled) Serial.println("Receiving bytes");
      for (int i = 0; i < bytesCount; i++) {
        waitForModemBytes(1, "Waiting for byte " + String(i+1));
        bytesPtr[i] = modem.read();
      }
      modem.write((uint8_t*) &bytesCount, 2);
      if (debugEnabled) Serial.println("Modem received " + String(bytesCount) + " bytes");
      return bytesCount;
    }

    digitalWrite(LEDG, LOW);
    delay(50);
    digitalWrite(LEDG, HIGH);
    delay(50);
    if (debugEnabled) Serial.println("Waiting for modem message");
  }

  if (debugEnabled) Serial.println("Get modem message timeout");
  return 0;
}


byte* defpointer;
int sendModemMessage(uint16_t recipient, uint16_t size, byte* bytes, bool expectResponse = false, byte*& response = defpointer) {
  bool received = false;

  while (!received) {
    if (debugEnabled) Serial.println("Attempting to send message");
    modem.write('s'); // Send command
    
    modem.write((uint8_t*) &recipient, 2);
    modem.write((uint8_t*) &size, 2);

    for (uint16_t i = 0; i < size; i++) {
      modem.write(bytes[i]);
      
      digitalWrite(LEDR, LOW);
      while(!modem.available()) {}
      digitalWrite(LEDR, HIGH);

      byte confirmation = modem.read();
      if (confirmation != bytes[i]) {
        digitalWrite(LEDR, LOW);
        while(true) {
          Serial.println("Sent byte #" + String(i) + " (" + String(bytes[i]) + ") to modem, confirmed " + String(confirmation));
          while (modem.available()) {
            Serial.print((char) modem.read());
            Serial.println();
          }
          delay(3000);
        }
      }
    }

    waitForModemBytes(2, "Waiting for message received confirmation");
    uint16_t count_confirmation;
    modem.readBytes((byte*) &count_confirmation, 2);
    if (count_confirmation != size) {
      digitalWrite(LEDR, LOW);
      while(true) {
        Serial.println("Sent " + String(size) + " bytes to modem, confirmed " + String(count_confirmation));
        delay(1000);
      }
    }

    if (expectResponse) {
      uint16_t ackSender; // TODO: Verify the sender is the wanted
      int responseCount = getModemMessage(response, ackSender);
      if (responseCount > 0) {
        return responseCount;
      }
    } else {
      return 0;
    }
  }
  return 0;
}


std::map<uint16_t, Metric> getNodeMetrics(std::vector<Node> nodes) {
  std::map<uint16_t, Metric> res;
  lock_modem = true;
  for (uint i = 0; i < nodes.size(); i++) {
    byte data[1] = {'n'};
    byte* response;
    int responseLength = sendModemMessage(nodes[i].address, 1, data, true, response);
    uint16_t amount;
    std::memcpy(&amount, response, sizeof(uint16_t));
    free(response);
    Metric metric;
    metric.num_epochs = amount;
    metric.hops = nodes[i].hops;
    res[nodes[i].address] = metric;
  }
  lock_modem = false;
  return res;
}

std::vector<float> requestWeights(uint16_t node, int batchNum, uint8_t scaled_weights_bits) {
  lock_modem = true;

  // Send a 'g' to the other devices so they start sending me their data
  byte data[4] = {'g', 0, 0, scaled_weights_bits}; // command, batch num (2), scaled_weights_bits
  std::memcpy(&data[1], &batchNum, sizeof(uint16_t));
  byte* response;
  int response_bytes = sendModemMessage(node, sizeof(data), data, true, response);

  float min_w, max_w;
  memcpy(&min_w, &response[0], sizeof(float));
  memcpy(&max_w, &response[sizeof(float)], sizeof(float));
  memcpy(&scaled_weights_bits, &response[sizeof(float) * 2], sizeof(uint8_t));

  std::vector<float> weights;
  uint currentResponseBit = sizeof(float) * 2 * 8; // After 2 floats
  scaledType weight = 0;
  for (uint byte = 0; byte < response_bytes - (sizeof(float) * 2); byte++) {
    for (uint bit = 0; bit < scaled_weights_bits; bit++) {
      uint bitValue = (response[currentResponseBit/8] >> 7 - (currentResponseBit % 8)) & 1;
      weight = weight << 1;
      weight |= bitValue;
      currentResponseBit++;
    }
    if (debugEnabled && byte == 0) Serial.println("Received first weight: " + String(weight));
    weights.push_back(deScaleWeight(min_w, max_w, weight, scaled_weights_bits));
    weight = 0;
  }
  free(response);
  lock_modem = false;
  return weights;
}

// Map containing the addres of the node and the samples it captured until the last FL
std::map<uint16_t, uint16_t> samples_amt;

void doFL(uint16_t target, uint8_t scaled_weights_bits) {
  digitalWrite(LEDB, LOW);
  Serial.println("Starting FL");

  if (scaled_weights_bits == 0) {
    scaled_weights_bits = 12;
  }

  std::vector<Node> nodes = getRoutingTable();
  Serial.println(nodes.size());
  if (!nodes.size()) {
    digitalWrite(LEDB, HIGH);
    return;
  }

  uint16_t best_node = 0;
  uint16_t max_epochs_since_last_fl = 0;
  uint16_t max_epochs = 0;
  std::map<uint16_t, Metric> node_metrics = getNodeMetrics(nodes);
  for (auto const& [node_addr, metric] : node_metrics) {
      uint16_t amount_since_last_fl = metric.num_epochs - samples_amt[node_addr];
      if ((target != 0 && node_addr == target) || (target == 0 && amount_since_last_fl >= max_epochs_since_last_fl)) {
        best_node = node_addr;
        max_epochs_since_last_fl = amount_since_last_fl;
        max_epochs = metric.num_epochs;
      }
  }

  Serial.println(max_epochs_since_last_fl);
  if (max_epochs_since_last_fl == 0) {
    digitalWrite(LEDB, HIGH);
    return;
  }

  samples_amt[best_node] = max_epochs;

  // Update local model (weighted average)
  float localWeightFactor = num_epochs / (float)(num_epochs + max_epochs_since_last_fl);
  float externalWeightFactor = max_epochs_since_last_fl / (float)(num_epochs + max_epochs_since_last_fl);
  Serial.println(localWeightFactor);
  Serial.println(externalWeightFactor);

  float* hidden_weights = network->getHiddenWeights();
  float* output_weights = network->getOutputWeights();
  std::vector<float> weights;

  int batches = floor((float) (network->getHiddenWeightsAmt() + network->getOutputWeightsAmt()) / (float) WEIGHTS_BATCH_SIZE);
  Serial.println(batches);

  for (uint16_t batchNum = 0; batchNum < batches; batchNum++) {
    std::vector<float> weights = requestWeights(best_node, batchNum, scaled_weights_bits);
    for(uint i = 0; i < weights.size(); i++) {
      uint weightPos = (batchNum * WEIGHTS_BATCH_SIZE) + i;
      if (weightPos < network->getHiddenWeightsAmt()) {
        hidden_weights[weightPos] = hidden_weights[weightPos] * localWeightFactor + weights[i] * externalWeightFactor;
      } else {
        weightPos = weightPos - network->getHiddenWeightsAmt();
        output_weights[weightPos] = output_weights[weightPos] * localWeightFactor + weights[i] * externalWeightFactor;
      }
    }
  }
  
  num_epochs += max_epochs_since_last_fl;

  delay(100);
  digitalWrite(LEDB, HIGH);
  Serial.println("FL_DONE");
}


void sendWeights(uint16_t recipient, uint16_t batchNum, uint8_t scaled_weights_bits) {
  if (debugEnabled) Serial.println("Received weights request for batch " + String(batchNum) + " from " + String(recipient));
  float min_w, max_w;
  // uint8_t scaled_weights_bits = 12;
  std::vector<scaledType> weights = getScaledWeights(batchNum, min_w, max_w, scaled_weights_bits);
  if (debugEnabled) Serial.println("Received " + String(weights.size()) + " weights from getScaledWeights, batch size is: " + String(weights.size()));

  uint num_bytes = sizeof(float) * 2 + sizeof(uint8_t) + std::ceil((weights.size() * scaled_weights_bits / 8.0));

  if (debugEnabled) Serial.println("Preparing " + String(num_bytes) + " to send");

  // The first 8 bytes will be the min and max weights, the rest will be the quantized weights
  byte data[num_bytes] = {}; // Initialized to zeros
  memcpy(&data, &min_w, sizeof(float));
  memcpy(&data[sizeof(float)], &max_w, sizeof(float));
  memcpy(&data[sizeof(float) * 2], &scaled_weights_bits, sizeof(float));

  if (debugEnabled) Serial.println("Min weight: " + String(min_w) + " Max weight: " + String(max_w));

  uint currentBit = sizeof(float) * 2 * 8;
  for(uint i = 0; i < weights.size(); i++) {
    if (i == 0 && debugEnabled) Serial.println("First weight: " + String(weights[i]));
    for (uint j = 0; j < scaled_weights_bits; j++) {
      uint shiftBits = scaled_weights_bits - j - 1;
      uint bitValue = (weights[i] >> shiftBits) & 1;
      data[currentBit / 8] |= bitValue << (7 - (currentBit % 8));
      currentBit += 1;
    }
  }

  sendModemMessage(recipient, num_bytes, data);
}

void sendMetrics(uint16_t recipient) {
  if (debugEnabled) Serial.println("Received samples count request, returning " + String(num_epochs));
  byte data[2];
  std::memcpy(&data, &num_epochs, sizeof(uint16_t));
  int responseLength = sendModemMessage(recipient, sizeof(uint16_t), data);
}

void setup() {
  pinMode(LEDR, OUTPUT);
  pinMode(LEDG, OUTPUT);
  pinMode(LEDB, OUTPUT);
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LEDR, HIGH);
  digitalWrite(LEDG, HIGH);
  digitalWrite(LEDB, HIGH);
  digitalWrite(LED_BUILTIN, HIGH);
 
  Serial.begin(115200);
  Serial1.begin(4800);

  network->initialize(0.6, 0.9);
}

float readSerialFloat() {
  float val;
  while (Serial.available() < 4) {}
  Serial.readBytes((byte*) &val, 4);
  return val;
}

void modemMessageAvailable() {
  if (debugEnabled) Serial.println("Modem message is available!");
  byte* bytes;
  uint16_t recipient;
  int bytesCount = getModemMessage(bytes, recipient);

  if (debugEnabled) Serial.println("Received " + String(bytesCount) + " bytes from modem");
  if (bytesCount == 4 && (char) bytes[0] == 'g') {
    uint16_t batchNum;
    std::memcpy(&batchNum, &bytes[1], sizeof(int16_t));
    uint8_t scaled_weights_bits;
    std::memcpy(&scaled_weights_bits, &bytes[3], sizeof(int8_t));
    sendWeights(recipient, batchNum, scaled_weights_bits);
  } else if (bytesCount == 1 && (char) bytes[0] == 'n') { // Amount of new samples request
    sendMetrics(recipient);
  }
}

void loop() {
  if (Serial.available()) {
    int read = Serial.read();
    if (read == '>') {          // s -> FEDERATED LEARNING
      uint16_t target;
      Serial.readBytes((byte*) &target, 2);

      uint8_t scaled_weights_bits;
      Serial.readBytes((byte*) &scaled_weights_bits, 1);

      doFL(target, scaled_weights_bits);
    } else if (read == 't') {   // Train with a sample
      trainWithSerialSample();
    } else if (read == 'r') {   // Get routing table
      Serial.println("Requesting routing table");
      std::vector<Node> nodes = getRoutingTable();
      Serial.println("Nodes: " + String(nodes.size()));
      for(uint i = 0; i < nodes.size(); i++) {
        Serial.println(nodes[i].address);
      }
    } else if (read == 'z') {   // Get weights
      float* hidden_weights = network->getHiddenWeights();
      float* output_weights = network->getOutputWeights();
      for (uint i = 0; i < network->getHiddenWeightsAmt(); i++) {Serial.write(hidden_weights[i]);}
      for (uint i = 0; i < network->getOutputWeightsAmt(); i++) {Serial.write(output_weights[i]);}
    } else if (read == 'x') {   // Get epochs
      Serial.println(num_epochs);
    } else if (read == 'i') {   // Initialize device
      Serial.println("Initializing device");

      network->initialize(0.6, 0.9);

      float* myHiddenWeights = network->getHiddenWeights();
      float* myOutputWeights = network->getOutputWeights();
      for (uint i = 0; i < network->getHiddenWeightsAmt(); i++) {
        myHiddenWeights[i] = readSerialFloat();
      }
      for (uint i = 0; i < network->getOutputWeightsAmt(); i++) {
        myOutputWeights[i] = readSerialFloat();
      }
      Serial.println("Model received");
    }
    
    // Modem message available
    if (useSerialModemPassthrough && read == 'm') {
      Serial.println("Reading modem message");
      modemMessageAvailable();
    }
  }
  
  if (!useSerialModemPassthrough && !lock_modem && modem.available()) { // Mesagge from LoRaMesher
    modemMessageAvailable();
  }
}
