#include "Arduino.h"
#include "RPC.h"


#define SERIAL_BR 115200
#define EIDSP_QUANTIZE_FILTERBANK   0
#include <training_kws_inference.h>

#include "neural_network.h"

#include <map>

int batchSize = 200;


// This variable will hold the recorded audio.
// Ideally this would only hold the features extracted in this core, but we have to support serial training on the M7, and that would require that bot cores knew how to extract the features
struct sharedMemory {
  // int var; // Required when a struct contains a flexible array member
  int16_t audio_input_buffer[16000];
  // weightType weights[hiddenWeightsAmt+outputWeightsAmt];
};
/* Datasheet: https://www.st.com/resource/en/reference_manual/dm00176879-stm32h745-755-and-stm32h747-757-advanced-arm-based-32-bit-mcus-stmicroelectronics.pdf
   In this region (SRAM3) we can use 32767 bytes (32kB)
   "AHB SRAM3 is mapped at address 0x3004 0000 and accessible by all system
    masters except BDMA through D2 domain AHB matrix. AHB SRAM3 can be used
    as buffers to store peripheral input/output data for Ethernet and USB, or as shared
    memory between the two cores."
*/
volatile struct sharedMemory * const shared_ptr = (struct sharedMemory *)0x30040000;

int getBatchSize(int batchNum) {
  if ((batchNum+1) * batchSize > hiddenWeightsAmt + outputWeightsAmt) { // Last batch is not full
    return hiddenWeightsAmt + outputWeightsAmt - (batchNum * batchSize);
  }
  return batchSize;
}





#ifdef CORE_CM7

#include <mbed.h>
#include "neural_network.h"
#include <map>
#include <vector>

static NeuralNetwork myNetwork;

uint16_t num_epochs = 0;

// Map containing the addres of the node and the samples it captured until the last FL
std::map<uint16_t, uint16_t> samples_amt;

std::vector<weightType> getWeightsValues(uint16_t batchNum) {
  weightType* myHiddenWeights = myNetwork.get_HiddenWeights();
  weightType* myOutputWeights = myNetwork.get_OutputWeights();

  std::vector<weightType> weights;
  for(int i = batchNum * batchSize; i < (batchNum+1)*batchSize; i++) {
    if (i > hiddenWeightsAmt+outputWeightsAmt) break;
    if (i < hiddenWeightsAmt) {
      weights.push_back(myHiddenWeights[i]);
    } else {
      weights.push_back(myOutputWeights[i-hiddenWeightsAmt]);
    }
  }
  Serial.println("[M7] Sending batch " + String(batchNum) + ". Weights: " + String(weights.size()));
  return weights;
}

// Check the RPC buffer and print it to serial
void proxy_serial_m4() {
  while(true) {
    while (RPC.available()) {
      //Serial.write(RPC.read());
      RPC.read();
    }
    rtos::ThisThread::sleep_for(50);
  }
}

// I don't like this cast to non-volatile...
static int get_input_data(size_t offset, size_t length, float *out_ptr) {
  numpy::int16_to_float((const int16_t*)&shared_ptr->audio_input_buffer[offset], out_ptr, length);
  return 0;
}

void train(int nb, bool only_forward) {
  Serial.println("LOG_START");

  signal_t signal;
  signal.total_length = EI_CLASSIFIER_RAW_SAMPLE_COUNT;
  signal.get_data = &get_input_data;
  ei::matrix_t features_matrix(1, EI_CLASSIFIER_NN_INPUT_FRAME_SIZE);

  EI_IMPULSE_ERROR r = get_one_second_features(&signal, &features_matrix, false);
  if (r != EI_IMPULSE_OK) {
    ei_printf("ERR: Failed to get features (%d)\n", r);
    return;
  }

  float myTarget[3] = {0};
  myTarget[nb - 1] = 1.f; // button 1 -> {1,0,0};  button 2 -> {0,1,0};  button 3 -> {0,0,1}

  // FORWARD
  unsigned long start = micros();
  float forward_error = myNetwork.forward(features_matrix.buffer, myTarget);
  Serial.println("Millis: " + String(micros()-start));

  float backward_error = 0;
  if (!only_forward) {
    // BACKWARD
    backward_error = myNetwork.backward(features_matrix.buffer, myTarget);
    num_epochs++;
  }

  float error = forward_error;

  float* myOutput = myNetwork.get_output();

  Serial.println("LOG_END");

  // Info to plot & graph!
  Serial.println("graph");

  // Print outputs
  for (size_t i = 0; i < 3; i++) {
    ei_printf_float(myOutput[i]);
    Serial.print(" ");
  }
  Serial.print("\n");

  // Print error
  ei_printf_float(error);
  Serial.print("\n");

  Serial.println(num_epochs, DEC);

  char* myError = (char*) &error;
  Serial.write(myError, sizeof(float));

  Serial.println(nb, DEC);
}

uint16_t getNewSamplesCount() {
  return num_epochs;
}

rtos::Thread thread;
void setup_m7() {
  Serial.begin(4800);
  myNetwork.initialize(0.6, 0.9, 0);
  RPC.bind("train", train);
  RPC.bind("getNewSamplesCount", getNewSamplesCount);
  RPC.bind("getWeightsValues", getWeightsValues);

  thread.start(proxy_serial_m4);
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

void doFL() {
  digitalWrite(LEDB, LOW);
  Serial.println("[M7] Starting FL");

  std::vector<uint16_t> nodes = RPC.call("getRoutingTable").as<std::vector<uint16_t>>();
  if (!nodes.size()) {
    Serial.println("[M7] Empty routing table");
    digitalWrite(LEDB, HIGH);
    return;
  }  else {
    Serial.println(String(nodes.size()) + " nodes");
  }

  uint16_t node = 0;
  uint16_t max_epochs_since_last_fl = 0;
  uint16_t max_epochs = 0;

  Serial.println("[M7] Getting nodes epochs");
  std::map<uint16_t, uint16_t> nodeEpochs = RPC.call("getNodesEpochs").as<std::map<uint16_t, uint16_t>>();
  Serial.println("[M7] Got nodes epochs");
  for (auto const& [nodeAddr, amount] : nodeEpochs) {
      uint16_t amount_since_last_fl = amount - samples_amt[nodeAddr];
      Serial.println("[M7] Device " + String(nodeAddr) + " captured " + String(amount) + " samples, " + amount_since_last_fl + " since last FL");
      if (amount_since_last_fl >= max_epochs_since_last_fl) {
        node = nodeAddr;
        max_epochs_since_last_fl = amount_since_last_fl;
        max_epochs = amount;
      }
  }

  if (max_epochs_since_last_fl == 0) {
    Serial.println("[M7] There are no new samples on any device");
    digitalWrite(LEDB, HIGH);
    return;
  }

  samples_amt[node] = max_epochs;

  // Update local model (weighted average)
  float localWeightFactor = num_epochs / (float)(num_epochs + max_epochs_since_last_fl);
  float externalWeightFactor = max_epochs_since_last_fl / (float)(num_epochs + max_epochs_since_last_fl);

  Serial.println("[M7] Local num epochs: " + String(num_epochs) + ". External: " + String(max_epochs_since_last_fl));
  Serial.println("[M7] Local weight factor: " + String(localWeightFactor) + ". External: " + String(externalWeightFactor));
  
  weightType* myHiddenWeights = myNetwork.get_HiddenWeights();
  weightType* myOutputWeights = myNetwork.get_OutputWeights();
  

  // -------------------
  Serial.println("READY");
  
  std::vector<weightType> weights;

  int batches = floor((float)(hiddenWeightsAmt + outputWeightsAmt) / (float)batchSize);
  for (uint16_t batchNum = 0; batchNum < batches; batchNum++) {
    // Serial.println("[M7] Requesting batch " + String(batchNum));
    std::vector<weightType> weights = RPC.call("getNodeWeights", node, batchNum).as<std::vector<weightType>>();
    for(int i = 0; i < weights.size(); i++) {
      int weightPos = (batchNum * batchSize) + i;
      if (weightPos < hiddenWeightsAmt) {
        myHiddenWeights[weightPos] = myHiddenWeights[weightPos] * localWeightFactor + weights[i] * externalWeightFactor;
      } else {
        weightPos = weightPos - hiddenWeightsAmt;
        myOutputWeights[weightPos] = myOutputWeights[weightPos] * localWeightFactor + weights[i] * externalWeightFactor;
      }
    }
  }
  
  num_epochs += max_epochs_since_last_fl;

  delay(1000);
  Serial.println("[M7] FL Done FL Done FL Done FL DoneFL Done FL Done FL Done FL Done FL Done FL Done");
  digitalWrite(LEDB, HIGH);
}

void loop_m7() {
  if (Serial.available()) {
    int read = Serial.read();
    if (read == '>') {          // s -> FEDERATED LEARNING
      doFL();
    } else if (read == 't') {   // Train with a sample
      trainWithSerialSample();
    } else if (read == 'r') {
      // Serial.println("[M7] Requesting routing table to M4");
      std::vector<uint16_t> nodes = RPC.call("getRoutingTable").as<std::vector<uint16_t>>();
      Serial.println("Nodes: " + String(nodes.size()));
      for(int i = 0; i < nodes.size(); i++) {
        Serial.println(nodes[i]);
      }
    } else if (read == 'g') {
      /*while (!RPC.available()) {}
      char num_button = RPC.read();
      record(num_button - (int)'0', false);*/
    } else if (read == 'z') {
      weightType* myHiddenWeights = myNetwork.get_HiddenWeights();
      weightType* myOutputWeights = myNetwork.get_OutputWeights();
      for (int i = 0; i < hiddenWeightsAmt; i++) {Serial.write(myHiddenWeights[i]);}
      for (int i = 0; i < outputWeightsAmt; i++) {Serial.write(myOutputWeights[i]);}
    } else if (read == 'x') {
      Serial.println(num_epochs);
    }
  }
  
  // proxy_serial_m4();
}

void ei_printf(const char *format, ...) {
  static char print_buf[1024] = { 0 };

  va_list args;
  va_start(args, format);
  int r = vsnprintf(print_buf, sizeof(print_buf), format, args);
  va_end(args);

  if (r > 0) {
    Serial.write(print_buf);
  }
}

#endif




#ifdef CORE_CM4

// #include <PDM.h>

static signed short sampleBuffer[2048]; // PDM sample buffer

/** Structure to manage audio capture */
typedef struct {
  uint8_t buf_ready;
  uint32_t buf_count;
  uint32_t n_samples;
} recording_t;

static recording_t recording = {0, 0, EI_CLASSIFIER_RAW_SAMPLE_COUNT};

volatile bool lock_modem = false;

std::vector<uint16_t> getRoutingTable() {
  lock_modem = true;
  std::vector<uint16_t> nodes;
  RPC.println("[M4] Getting routing table");
  Serial1.write('r'); // Send command
  while (!Serial1.available()) {
    RPC.println("[M4] Waiting for nodes count");
    delay(500);
  }
  uint8_t count = Serial1.read();
  RPC.println("[M4] Number of nodes: " + String(count));
  for (int i = 0; i < count; i++) {
    while (Serial1.available() < 2) {
      RPC.println("[M4] Waiting for node address");
      delay(500);
    }
    uint16_t node;
    Serial1.readBytes((char*)&node, 2);
    nodes.push_back(node);
    RPC.println("[M4] Node " + String(i + 1) + ": " + String(nodes[i]));
  }
  lock_modem = false;
  return nodes;
}

int getModemMessage(byte*& bytesPtr, uint16_t &recipient) {
  int limit = millis() + 30000;
  while (millis() < limit) {
    RPC.println("[M4] Waiting for modem message"); delay(1000);
    if (Serial1.available()) {
      char command = Serial1.read();
      RPC.println("[M4] Recevied command: " + String(command));
      while (Serial1.available() < 2) {
        RPC.println("[M4] Waiting for recipient");
        delay(100);
      }

      // Read from
      byte addressBytes[2];
      Serial1.readBytes(addressBytes, 2);
      memcpy(&recipient, addressBytes, sizeof(uint16_t));

      while (Serial1.available() < 2) {
        RPC.println("[M4] Waiting for bytes count");
        delay(100);
      }

      // Read message bytes count
      byte countBytes[2];
      uint16_t bytesCount;
      Serial1.readBytes(countBytes, 2);
      memcpy(&bytesCount, countBytes, sizeof(uint16_t));

      RPC.println("[M4] Bytes count: " + String(bytesCount));
      bytesPtr = (byte*) malloc(bytesCount);
      RPC.println("Receiving bytes");
      for (int i = 0; i < bytesCount; i++) {
        while (!Serial1.available()) { RPC.println("[M4] Waiting for byte " + String(i+1)); delay(100); }
        bytesPtr[i] = Serial1.read();
        RPC.println("[M4] Got byte " + String(i + 1) + " (int): " + String((int)bytesPtr[i]));
      }
      RPC.print("[M4] Modem received " + String(bytesCount) + " bytes");
      return bytesCount;
    }
  }

  RPC.println("[M4] Get modem message timeout");
  return 0;
}

byte* defpointer;
int sendModemMessage(uint16_t recipient, uint16_t size, byte* bytes, bool expectResponse = false, byte*& response = defpointer) {
  bool received = false;
  while (!received) {
    RPC.println("[M4] Attempting to send message");
    Serial1.write('s'); // Send command
    Serial1.write(static_cast<byte*>(static_cast<void*>(&recipient)), 2);
    Serial1.write(static_cast<byte*>(static_cast<void*>(&size)), 2);
    for (uint16_t i = 0; i < size; i++) {
      Serial1.write(bytes[i]);
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

std::map<uint16_t, uint16_t> getNodesEpochs() {
  std::map<uint16_t, uint16_t> res;
  std::vector<uint16_t> nodes = getRoutingTable();
  lock_modem = true;
  for (int i = 0; i < nodes.size(); i++) {
    byte data[1] = {'n'};
    byte* response;
    RPC.println("Sending message requesting epochs");
    int responseLength = sendModemMessage(nodes[i], 1, data, true, response);
    uint16_t amount;
    std::memcpy(&amount, response, sizeof(uint16_t));
    res[nodes[i]] = amount;
  }
  lock_modem = false;
  return res;
}

std::vector<weightType> getNodeWeights(uint16_t node, int batchNum) {
  lock_modem = true;
  RPC.print("[M4] Batch "); RPC.print(batchNum); RPC.println(" requested");

  // Send a 'g' to the other devices so they start sending me their data
  byte data[3] = {'g', 0, 0};
  std::memcpy(&data[1], &batchNum, sizeof(uint16_t));
  byte* response;
  int responseLength = sendModemMessage(node, 3, data, true, response);

  RPC.println("[M4] Batch " + String(batchNum) + " received, responseLength: "+ String(responseLength));
  std::vector<weightType> weights;
  for (int k = 0; k < responseLength; k++) {
    weights.push_back(response[k]);
    // TODO: In this new version only byte weighttype is supported
  }
  lock_modem = false;
  return weights;
}

void sendNewSamplesCount(uint16_t recipient) {
  uint16_t num_epochs = RPC.call("getNewSamplesCount").as<uint16_t>();

  RPC.println("[M4] Received samples count request, returning " + String(num_epochs));
  byte data[2];
  std::memcpy(&data, &num_epochs, sizeof(uint16_t));
  int responseLength = sendModemMessage(recipient, sizeof(uint16_t), data);
}

void sendWeights(uint16_t recipient, uint16_t batchNum) {
  RPC.println("[M4] Received weights request for batch : " + String(batchNum));
  std::vector<weightType> weights = RPC.call("getWeightsValues", batchNum).as<std::vector<weightType>>();
  RPC.println("[M4] Recevied " + String(weights.size()) + " weights from getWeightsValues, batch size is: " + String(getBatchSize(batchNum)));
  byte weightsArr[batchSize];
  for(int i = 0; i < weights.size(); i++) {
    weightsArr[i] = (byte) weights[i];
  }
  sendModemMessage(recipient, getBatchSize(batchNum), weightsArr);
  RPC.println("[M4] Batch sent!");
}

void setup_m4() {
  Serial1.begin(4800);
  
  RPC.bind("getRoutingTable", getRoutingTable);
  RPC.bind("getNodesEpochs", getNodesEpochs);
  RPC.bind("getNodeWeights", getNodeWeights);
  /*if (microphone_setup() == false) {
    ei_printf("ERR: Failed to setup audio sampling\r\n");
    return;
  }*/
}

void loop_m4() {
  if (!lock_modem && Serial1.available()) { // Mesagge from LoRaMesher
    RPC.println("[M4] Modem message is available!");
    byte* bytes;
    uint16_t recipient;
    int bytesCount = getModemMessage(bytes, recipient);

    RPC.println("[M4] Received " + String(bytesCount) + " bytes from modem");
    if (bytesCount == 3 && (char) bytes[0] == 'g') {
      uint16_t batchNum;
      std::memcpy(&batchNum, &bytes[1], sizeof(int16_t));
      sendWeights(recipient, batchNum);
    } else if (bytesCount == 1 && (char) bytes[0] == 'n') { // Amount of new samples request
      sendNewSamplesCount(recipient);
    }
  }
}

#endif







void setup() {
  pinMode(LEDR, OUTPUT);
  pinMode(LEDG, OUTPUT);
  pinMode(LEDB, OUTPUT);
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LEDR, HIGH);
  digitalWrite(LEDG, HIGH);
  digitalWrite(LEDB, HIGH);
  digitalWrite(LED_BUILTIN, HIGH);
 

  RPC.begin();
  #ifdef CORE_CM7
    setup_m7();
  #else
    setup_m4();
  #endif
}


void loop() {
  #ifdef CORE_CM7
    loop_m7();
  #else
    loop_m4();
  #endif
}
