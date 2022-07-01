#ifdef CORE_CM7

#include <mbed.h>


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
      Serial.write(RPC.read());
    }
    rtos::ThisThread::sleep_for(50);
  }
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


void loop_m7() {
  if (Serial.available()) {
    int read = Serial.read();
    if (read == '>') {          // s -> FEDERATED LEARNING
      doFL();
    } else if (read == 't') {   // Train with a sample
      trainWithSerialSample();
    } else if (read == 'r') {
      std::vector<uint16_t> nodes = RPC.call("getRoutingTable").as<std::vector<uint16_t>>();
    } else if (read == 'g') {
      /*while (!RPC.available()) {}
      char num_button = RPC.read();
      record(num_button - (int)'0', false);*/
    }
  }
  
  // proxy_serial_m4();
}




uint16_t getNewSamplesCount() {
  return num_epochs;
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
  int batches = floor((float)(hiddenWeightsAmt + outputWeightsAmt) / (float)batchSize);
  for (uint16_t batchNum = 0; batchNum < batches; batchNum++) {
    Serial.println("[M7] Requesting weights for batch " + String(batchNum) + " / " + String(batches));
    std::vector<weightType> weights = RPC.call("getNodeWeights", node, batchNum).as<std::vector<weightType>>();
    Serial.println("[M7] Got " + String(weights.size()) + " weights for batch " + String(batchNum));
    for(int i = 0; i < weights.size(); i++) {
      int weightPos = (batchNum * batchSize) + i;
      if (weightPos < hiddenWeightsAmt) {
        Serial.println("[M7] Received weight " + String(weightPos) + ": " + String(weights[i]) + ", Local weight: " + String(myHiddenWeights[weightPos]) + ", Result: " + String(myHiddenWeights[weightPos] * localWeightFactor + weights[i] * externalWeightFactor));
        myHiddenWeights[weightPos] = myHiddenWeights[weightPos] * localWeightFactor + weights[i] * externalWeightFactor;
      } else {
        weightPos = weightPos - hiddenWeightsAmt;
        Serial.println("[M7] Received weight " + String(weightPos) + ": " + String(weights[i]) + ", Local weight: " + String(myOutputWeights[weightPos]) + ", Result: " + String(myOutputWeights[weightPos] * localWeightFactor + weights[i] * externalWeightFactor));
        myOutputWeights[weightPos] = myOutputWeights[weightPos] * localWeightFactor + weights[i] * externalWeightFactor;
      }
    }
  }

  num_epochs += max_epochs_since_last_fl;

  delay(1000);
  Serial.println("[M7] FL Done FL Done FL Done FL DoneFL Done FL Done FL Done FL Done FL Done FL Done");
  digitalWrite(LEDB, HIGH);
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

// I don't like this cast to non-volatile...
static int get_input_data(size_t offset, size_t length, float *out_ptr) {
  numpy::int16_to_float((const int16_t*)&shared_ptr->audio_input_buffer[offset], out_ptr, length);
  return 0;
}



#endif
