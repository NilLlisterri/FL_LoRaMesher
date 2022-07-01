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
