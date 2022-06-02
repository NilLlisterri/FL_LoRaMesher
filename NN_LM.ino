#define EIDSP_QUANTIZE_FILTERBANK   0
#include <training_kws_inference.h>
#include "neural_network.h"
#include <map>
#include <PDM.h>

/** Audio buffers, pointers and selectors */
typedef struct {
  int16_t buffer[EI_CLASSIFIER_RAW_SAMPLE_COUNT];
  uint8_t buf_ready;
  uint32_t buf_count;
  uint32_t n_samples;
} inference_t;

static inference_t inference;
static bool debug_nn = false; // Set this to true to see e.g. features generated from the raw signal


const uint8_t button_1 = 2;
const uint8_t button_2 = 3;
const uint8_t button_3 = 4;
const uint8_t button_4 = 5;
//uint8_t num_button = 0; // 0 represents none
bool button_pressed = false;

int batchSize = 200;

std::map<uint16_t, uint16_t> samples_amt;

static NeuralNetwork myNetwork;

uint16_t num_epochs = 0;

void setup() {
  Serial.begin(4800);
  Serial1.begin(4800);
  
  pinMode(button_1, INPUT);
  pinMode(button_2, INPUT);
  pinMode(button_3, INPUT);
  pinMode(button_4, INPUT);
  pinMode(LED_BUILTIN, OUTPUT);
  pinMode(LEDR, OUTPUT);
  pinMode(LEDG, OUTPUT);
  pinMode(LEDB, OUTPUT);
  digitalWrite(LEDR, HIGH);
  digitalWrite(LEDG, HIGH);
  digitalWrite(LEDB, HIGH);

  digitalWrite(LED_BUILTIN, HIGH);

  if (microphone_setup(EI_CLASSIFIER_RAW_SAMPLE_COUNT) == false) {
    ei_printf("ERR: Failed to setup audio sampling\r\n");
    return;
  }

  myNetwork.initialize(0.6, 0.9, 0);
  digitalWrite(LED_BUILTIN, LOW);
}

std::vector<uint16_t> getRoutingTable() {
  std::vector<uint16_t> nodes;
  Serial.println("Getting routing table");
  Serial1.write('r'); // Send command
  while(!Serial1.available()) {
    Serial.println("Waiting for modem response");
    delay(500);
  }
  uint8_t count = Serial1.read();
  Serial.print("Number of nodes: "); Serial.println(count);
  for(int i = 0; i < count; i++) {
    while(Serial1.available() < 2) {
      Serial.println("Waiting for modem response");
      delay(500);
    }
    uint16_t node;
    Serial1.readBytes((char*)&node, 2);
    nodes.push_back(node);
    Serial.println("Node " + String(i+1) + ": " + String(nodes[i]));
  }
  return nodes;
}

int sendModemMessage(uint16_t recipient, uint16_t size, byte* bytes, bool expectResponse = false, byte* response = 0) {
  bool received = false;
  while(!received) {
    Serial.println("Attempting to send message");
    Serial1.write('s'); // Send command
    Serial1.write(static_cast<byte*>(static_cast<void*>(&recipient)), 2);
    Serial1.write(static_cast<byte*>(static_cast<void*>(&size)), 2);
    for(uint16_t i = 0; i < size; i++) {
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

void sendNewSamplesCount(uint16_t recipient) {
  byte data[2];
  uint16_t count = samples_amt[recipient];
  std::memcpy(&data, &count, sizeof(uint16_t));
  int responseLength = sendModemMessage(recipient, sizeof(uint16_t), data);
}

int getModemMessage(byte*& bytesPtr, uint16_t &recipient) {
  int limit = millis() + 30000;
  while(millis() < limit) {
    Serial.println(millis());
    Serial.println(limit);
    Serial.println("Waiting for modem message"); delay(1000);
    if(Serial1.available()) {
      char command = Serial1.read();
      Serial.print("Recevied command: "); Serial.println(command);
      while(Serial1.available() < 2) {Serial.println("Waiting for recipient"); delay(100);}

      // Read from
      byte addressBytes[2];
      Serial1.readBytes(addressBytes, 2);
      memcpy(&recipient, addressBytes, sizeof(uint16_t));

      while(Serial1.available() < 2) {Serial.println("Waiting for bytes count"); delay(100);}

      // Read message bytes count
      byte countBytes[2];
      uint16_t bytesCount;
      Serial1.readBytes(countBytes, 2);
      memcpy(&bytesCount, countBytes, sizeof(uint16_t));
      
      Serial.print("Bytes count: "); Serial.println(bytesCount);
      bytesPtr = (byte*) malloc(bytesCount);
      Serial.println("Receiving bytes");
      for(int i = 0; i < bytesCount; i++) {
        while(!Serial1.available()) {}
        bytesPtr[i] = Serial1.read();
        // Serial.print("Got byte "); Serial.println(i+1);
        // Serial.print((char) bytesPtr[i]);  
      }
      Serial.print("Modem received "); Serial.print(bytesCount); Serial.println(" bytes");
      return bytesCount;
    }
  }

  Serial.println("Get modem message timeout");
  return 0;
}



void sendWeights(uint16_t recipient, uint16_t batchNum) {
  Serial.print("Received weights request for batch : "); Serial.println(batchNum);
  weightType* myHiddenWeights = myNetwork.get_HiddenWeights();
  weightType* myOutputWeights = myNetwork.get_OutputWeights();

  samples_amt[recipient] = 0;

  byte weights[batchSize];
  int j = 0;
  for (int i = batchNum * batchSize; i < (batchNum+1) * batchSize && i < myNetwork.hiddenWeightsAmt + myNetwork.outputWeightsAmt; i++) {
    if (i < myNetwork.hiddenWeightsAmt) {
      weights[j] = myHiddenWeights[i];
    } else {
      weights[j] = myOutputWeights[i - myNetwork.hiddenWeightsAmt];
    }
    j++;
  }
  sendModemMessage(recipient, j, weights);
  Serial.println("Batch sent!");
}

int getBatchesAmt() {
  return ceil((float)(myNetwork.hiddenWeightsAmt + myNetwork.outputWeightsAmt) / (float)batchSize);
}


void doFL() {
  digitalWrite(LEDB, LOW);
  Serial.println("Starting FL");

 
  std::vector<uint16_t> nodes = getRoutingTable();
  if (!nodes.size()) {
      Serial.println("Empty routing table");
      return;
  }

  int max_i = 0;
  for(int i = 0; i < nodes.size(); i++) {
    byte data[1] = {'n'};
    byte* response;
    int responseLength = sendModemMessage(nodes[i], 1, data, true, response);
    uint16_t amount;
    std::memcpy(&amount, &response, sizeof(uint16_t));
    Serial.println("Device " + String(nodes[i]) + " captured " + String(amount) + " samples");
    if (amount >= max_i) max_i = i;
  }

  uint16_t node = nodes[max_i];
  
  weightType hw[myNetwork.hiddenWeightsAmt]; 
  weightType ow[myNetwork.outputWeightsAmt]; 

  uint16_t batches = getBatchesAmt();
  for (uint16_t i = 0; i < batches; i++) {
    Serial.print("Batch "); Serial.print(i); Serial.print("/"); Serial.print(batches); Serial.println(" requested");

     // Send a 'g' to the other devices so they start sending me their data
    byte data[3] = {'g', 0, 0};
    std::memcpy(&data[1], &i, sizeof(uint16_t));
    byte* response;
    int responseLength = sendModemMessage(node, 3, data, true, response);

    Serial.println("Batch received!");

    for(int k = 0; k < responseLength; k++) {
      int pos = i*batchSize+k;
      if (pos > myNetwork.hiddenWeightsAmt) {
        hw[pos] = response[k];
      } else {
        ow[pos - myNetwork.hiddenWeightsAmt] = response[k];
      }
    }
  }

  Serial.println("FL Done");
  digitalWrite(LED_BUILTIN, LOW);    // OFF
}


void trainWithSerialSample() {
  Serial.println("ok");

  while (Serial.available() < 1) {}
  uint8_t num_button = Serial.read();
  Serial.print("Button "); Serial.println(num_button);

  while (Serial.available() < 1) {}
  bool only_forward = Serial.read() == 1;
  Serial.print("Only forward "); Serial.println(only_forward);

  byte ref[2];
  for (int i = 0; i < EI_CLASSIFIER_RAW_SAMPLE_COUNT; i++) {
    while (Serial.available() < 2) {}
    Serial.readBytes(ref, 2);
    inference.buffer[i] = 0;
    inference.buffer[i] = (ref[1] << 8) | ref[0];
  }
  Serial.print("Sample received for button ");
  Serial.println(num_button);
  train(num_button, only_forward);
}

void doRecord(uint8_t num_button, bool only_forward) {
  digitalWrite(LEDR, LOW);

  Serial.println("Recording...");
  bool m = microphone_inference_record();
  if (!m) {
    Serial.println("ERR: Failed to record audio...");
    return;
  }
  Serial.println("Recording done");

  train(num_button, only_forward);

  button_pressed = false;
  digitalWrite(LEDR, HIGH);
}

float readFloat() {
  byte res[4];
  while (Serial.available() < 4) {}
  for (int n = 0; n < 4; n++) {
    res[n] = Serial.read();
  }
  return *(float *)&res;
}

void train(int nb, bool only_forward) {
  Serial.println("LOG_START");

  signal_t signal;
  signal.total_length = EI_CLASSIFIER_RAW_SAMPLE_COUNT;
  signal.get_data = &microphone_audio_signal_get_data;
  ei::matrix_t features_matrix(1, EI_CLASSIFIER_NN_INPUT_FRAME_SIZE);

  EI_IMPULSE_ERROR r = get_one_second_features(&signal, &features_matrix, debug_nn);
  if (r != EI_IMPULSE_OK) {
    ei_printf("ERR: Failed to get features (%d)\n", r);
    return;
  }

  float myTarget[3] = {0};
  myTarget[nb - 1] = 1.f; // button 1 -> {1,0,0};  button 2 -> {0,1,0};  button 3 -> {0,0,1}

  // FORWARD
  float forward_error = myNetwork.forward(features_matrix.buffer, myTarget);

  float backward_error = 0;
  if (!only_forward) {
    // BACKWARD
    backward_error = myNetwork.backward(features_matrix.buffer, myTarget);
    ++num_epochs;

    for (auto [node, amt] : samples_amt) {
      samples_amt[node]++;
    }
  }

  float error = forward_error;
  if (!only_forward) {
    // error = backward_error;
  }

  float* myOutput = myNetwork.get_output();

  //uint8_t num_button_output = 0;
  //float max_output = 0.f;
  // Serial.print("Inference result: ");

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

static void pdm_data_ready_inference_callback(void) {
  int bytesAvailable = PDM.available();

  // read into the sample buffer
  int bytesRead = PDM.read((char *)&sampleBuffer[0], bytesAvailable);

  if (inference.buf_ready == 0) {
      for(int i = 0; i < bytesRead>>1; i++) {
          inference.buffer[inference.buf_count++] = sampleBuffer[i];

          if(inference.buf_count >= inference.n_samples) {
              inference.buf_count = 0;
              inference.buf_ready = 1;
              break;
          }
      }
  }
}

static bool microphone_setup(uint32_t n_samples) {
  inference.buf_count  = 0;
  inference.n_samples  = n_samples;
  inference.buf_ready  = 0;

  // configure the data receive callback
  PDM.onReceive(&pdm_data_ready_inference_callback);

  // optionally set the gain, defaults to 20
  PDM.setGain(80);
  PDM.setBufferSize(4096);

  // initialize PDM with:
  // - one channel (mono mode)
  // - a 16 kHz sample rate
  if (!PDM.begin(1, EI_CLASSIFIER_FREQUENCY)) {
      ei_printf("Failed to start PDM!");
      PDM.end();
      return false;
  }

  return true;
}

static bool microphone_inference_record(void) {
  inference.buf_ready = 0;
  inference.buf_count = 0;
  while (inference.buf_ready == 0) {
    delay(10);
  }
  return true;
}

static int microphone_audio_signal_get_data(size_t offset, size_t length, float *out_ptr) {
  numpy::int16_to_float(&inference.buffer[offset], out_ptr, length);
  return 0;
}

void loop() {
  digitalWrite(LEDR, HIGH);           // OFF
  digitalWrite(LEDG, HIGH);           // OFF
  digitalWrite(LEDB, HIGH);           // OFF
  digitalWrite(LED_BUILTIN, HIGH);    // OFF
 
  int read = Serial.read();
  if (read == '>') {          // s -> FEDERATED LEARNING
    doFL();
  } else if (read == 't') {   // Train with a sample
    trainWithSerialSample();
  } else if (read == 'r') {
    getRoutingTable();
  } else if (read == 'g') {
    while(!Serial.available()) {delay(100);}
    char num_button = Serial.read();
    doRecord(num_button - (int)'0', false);
  }

  if (Serial1.available()) { // Mesagge from LoRaMesher
    byte* bytes;
    uint16_t recipient;
    int bytesCount = getModemMessage(bytes, recipient);
    
    Serial.print("Received "); Serial.print(bytesCount); Serial.println(" bytes from modem");
    if (bytesCount == 3 && (char) bytes[0] == 'g') {
      uint16_t batchNum;
      std::memcpy(&batchNum, &bytes[1], sizeof(int16_t));
      sendWeights(recipient, batchNum);
    } else if (bytesCount == 1 && (char) bytes[0] == 'n') { // Amount of new samples request
      sendNewSamplesCount(recipient);
    }
  }
}
