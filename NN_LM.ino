#define EIDSP_QUANTIZE_FILTERBANK   0
#include <training_kws_inference.h>
#include "neural_network.h"


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

// Defaults: 0.3, 0.9
static NeuralNetwork myNetwork;
const float threshold = 0.6;

uint16_t num_epochs = 0;

void setup() {
  Serial.begin(115200);
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

  num_epochs = 0;
}

void loop() {
  digitalWrite(LEDR, HIGH);           // OFF
  digitalWrite(LEDG, HIGH);           // OFF
  digitalWrite(LEDB, HIGH);           // OFF
  digitalWrite(LED_BUILTIN, HIGH);    // ON

  /*if (button_pressed == true) {
    doRecord();
  }*/

 
  int read = Serial.read();
  if (read == '>') {          // s -> FEDERATED LEARNING
    doFL();
  } else if (read == 't') {   // Train with a sample
    trainWithSerialSample();
  }
  // else if (read == 'r') {
  //  Serial1.write('r');
  //}

  if (Serial1.available()) { // Mesagge from LoRaMesher
    byte* bytes;
    int bytesCount = getModemMessage(bytes);
    Serial.print("Received "); Serial.print(bytesCount); Serial.println(" bytes from modem");
    if (bytesCount == 1 && (char) bytes[0] == 'g') {
      Serial.println("Received weights request");
      sendWeights();
    }
  }
}

int getModemMessage(byte*& bytesPtr) {
  while(!Serial1.available()) {Serial.println("Waiting for modem message"); delay(500);}
  char command = Serial1.read();
  Serial.print("Recevied command: "); Serial.println(command);
  while(!Serial1.available()) {Serial.println("Waiting for size"); delay(100);}
  byte ref[2];
  uint16_t bytesCount;
  Serial1.readBytes(ref, 2);
  memcpy(&bytesCount, ref, sizeof(uint16_t));
  //int bytesCount = Serial1.read(, 2);
  Serial.print("Bytes count: "); Serial.println(bytesCount);
  bytesPtr = (byte*) malloc(bytesCount);
  Serial.println("Receiving bytes");
  for(int i = 0; i < bytesCount; i++) {
    while(!Serial1.available()) {}
    bytesPtr[i] = Serial1.read();
    // Serial.print("Got byte "); Serial.println(i+1);
    // Serial.print((char) bytesPtr[i]);  
  }
  Serial.println("Bytes received");
  return bytesCount;
}

void sendWeights() {
  Serial.print("Sending "); Serial.print((InputNodes + 1) * HiddenNodes); Serial.println(" weights");
  int8_t* myHiddenWeights = myNetwork.get_HiddenWeights();
  Serial1.write('s');
  uint16_t size = (InputNodes + 1) * HiddenNodes;
  Serial1.write(static_cast<byte*>(static_cast<void*>(&size)), 2);
  for (int i = 0; i < (InputNodes + 1) * HiddenNodes; ++i) {
    Serial1.write(myHiddenWeights[i]);
  }

  delay(10000);
  Serial.print("Sending "); Serial.print((HiddenNodes + 1) * OutputNodes); Serial.println(" weights");
  int8_t* myOutputWeights = myNetwork.get_OutputWeights();
  Serial1.write('s');
  size = (InputNodes + 1) * HiddenNodes;
  Serial1.write(static_cast<byte*>(static_cast<void*>(&size)), 2);
  for (int i = 0; i < (HiddenNodes + 1) * OutputNodes; ++i) {
    Serial1.write(myOutputWeights[i]);
  }
}

void doFL() {
  digitalWrite(LEDB, LOW);    // ON
  
  /*while(true) {
    if (Serial1.available()) Serial.print((char)Serial1.read());
  
    if(Serial.available()) Serial1.print((char)Serial.read());
  }*/
  Serial.println("Starting FL");

  Serial.println("Getting input weights");

  Serial1.write('s'); // Send command
  uint16_t size = 1;
  Serial1.write(static_cast<byte*>(static_cast<void*>(&size)), 2);
  Serial1.write('g');  // Send a 'g' to the other devices so they start sending me their data

  Serial.println("Send command sent");

  byte* inputWeights;
  int bytesCount = getModemMessage(inputWeights);

  Serial.println("Getting output weights");

  byte* outputWeights;
  bytesCount = getModemMessage(outputWeights);
  
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
  Serial.println("Recording...");
  bool m = microphone_inference_record();
  if (!m) {
    Serial.println("ERR: Failed to record audio...");
    return;
  }
  Serial.println("Recording done");

  train(num_button, only_forward);

  button_pressed = false;
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
  weightType* myHiddenWeights = myNetwork.get_HiddenWeights();
  for (int i = 0; i < (InputNodes+1) * HiddenNodes; i = i+300) {
    Serial.print("Weight "); Serial.print(i); Serial.print(": "); Serial.println(myHiddenWeights[i]);
  }
  
  
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

static bool microphone_setup(uint32_t n_samples) {
  inference.buf_count  = 0;
  inference.n_samples  = n_samples;
  inference.buf_ready  = 0;

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
