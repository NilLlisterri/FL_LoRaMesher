// This code is a modification of the code from http://robotics.hobbizine.com/arduinoann.html

#include <arduino.h>
#include "neural_network.h"
#include <math.h>

void NeuralNetwork::initialize(float LearningRate, float Momentum, int DropoutRate) {
    this->LearningRate = LearningRate;
    this->Momentum = Momentum;
    this->DropoutRate = DropoutRate;

    for (int i = 0; i < (InputNodes+1) * HiddenNodes; ++i) {
      HiddenWeights[i] = setWeight(random(InitialWeightMin*100, InitialWeightMax*100)/100.f); // Random generates ints
    }

    for (int i = 0; i < (HiddenNodes+1) * OutputNodes; ++i) {
      OutputWeights[i] = setWeight(random(InitialWeightMin*100, InitialWeightMax*100)/100.f);
    }
}

float NeuralNetwork::forward(const float Input[], const float Target[]){
    float error = 0;

    /******************************************************************
    * Compute hidden layer activations
    ******************************************************************/
    for (int i = 0; i < HiddenNodes; i++) {
        float Accum = getWeight(HiddenWeights[InputNodes*HiddenNodes + i]);
        for (int j = 0; j < InputNodes; j++) {
            Accum += Input[j] * getWeight(HiddenWeights[j*HiddenNodes + i]);
        }
        Hidden[i] = 1.0 / (1.0 + exp(-Accum));
    }

    /******************************************************************
    * Compute output layer activations and calculate errors
    ******************************************************************/
    for (int i = 0; i < OutputNodes; i++) {
        float Accum = getWeight(OutputWeights[HiddenNodes*OutputNodes + i]);
        for (int j = 0; j < HiddenNodes; j++) {
            Accum += Hidden[j] * getWeight(OutputWeights[j*OutputNodes + i]);
        }
        Output[i] = 1.0 / (1.0 + exp(-Accum));
        // OutputDelta[i] = (Target[i] - Output[i]) * Output[i] * (1.0 - Output[i]);
        error += 0.33333 * (Target[i] - Output[i]) * (Target[i] - Output[i]);
    }
    return error;
}

float NeuralNetwork::getWeight(weightType val) {
  return val / weightFactor;
}

weightType NeuralNetwork::setWeight(float val) {
  return val * weightFactor;
}

// Input will be changed!!
float NeuralNetwork::backward(float Input[], const float Target[]){
    float error = 0;

    for (int i = 0; i < InputNodes; i++) {
        if (rand() % 100 < this->DropoutRate) {
            Input[i] = 0;
        }
    }

    // Forward
    /******************************************************************
    * Compute hidden layer activations
    ******************************************************************/
    for (int i = 0; i < HiddenNodes; i++) {
        float Accum = getWeight(HiddenWeights[InputNodes*HiddenNodes + i]);
        for (int j = 0; j < InputNodes; j++) {
            Accum += Input[j] * getWeight(HiddenWeights[j*HiddenNodes + i]);
        }
        Hidden[i] = 1.0 / (1.0 + exp(-Accum));
    }

    /******************************************************************
    * Compute output layer activations and calculate errors
    ******************************************************************/
    for (int i = 0; i < OutputNodes; i++) {
        float Accum = getWeight(OutputWeights[HiddenNodes*OutputNodes + i]);
        for (int j = 0; j < HiddenNodes; j++) {
            Accum += Hidden[j] * getWeight(OutputWeights[j*OutputNodes + i]);
        }
        Output[i] = 1.0 / (1.0 + exp(-Accum)); // Sigmoid, from 0 to 1
        OutputDelta[i] = (Target[i] - Output[i]) * Output[i] * (1.0 - Output[i]);
        /*Serial.print("OutputDelta "); Serial.print(i); Serial.print(": "); Serial.println(OutputDelta[i]);
        Serial.print("OutputDelta "); Serial.print(i); Serial.print(": "); Serial.println(OutputDelta[i]);
        Serial.print("OutputAccoum "); Serial.print(i); Serial.print(": "); Serial.println(Accum);*/
        error += 1/OutputNodes * (Target[i] - Output[i]) * (Target[i] - Output[i]);
    }
    // End forward

    // Backward
    /******************************************************************
    * Backpropagate errors to hidden layer
    ******************************************************************/
    for(int i = 0 ; i < HiddenNodes ; i++ ) {    
        float Accum = 0.0 ;
        for(int j = 0 ; j < OutputNodes ; j++ ) {
            Accum += getWeight(OutputWeights[i*OutputNodes + j]) * OutputDelta[j] ;
        }
        HiddenDelta[i] = Accum * Hidden[i] * (1.0 - Hidden[i]) ;
    }

    /******************************************************************
    * Update Inner-->Hidden Weights
    ******************************************************************/
    for(int i = 0 ; i < HiddenNodes ; i++ ) {     
        ChangeHiddenWeights[InputNodes*HiddenNodes + i] = setWeight(LearningRate * HiddenDelta[i] + Momentum * getWeight(ChangeHiddenWeights[InputNodes*HiddenNodes + i]));
        HiddenWeights[InputNodes*HiddenNodes + i] += ChangeHiddenWeights[InputNodes*HiddenNodes + i];
        for(int j = 0 ; j < InputNodes ; j++ ) { 
            ChangeHiddenWeights[j*HiddenNodes + i] = setWeight(LearningRate * Input[j] * HiddenDelta[i] + Momentum * getWeight(ChangeHiddenWeights[j*HiddenNodes + i]));
            HiddenWeights[j*HiddenNodes + i] += ChangeHiddenWeights[j*HiddenNodes + i];
        }
    }

    /******************************************************************
    * Update Hidden-->Output Weights
    ******************************************************************/
    for(int i = 0 ; i < OutputNodes ; i ++ ) {    
        ChangeOutputWeights[HiddenNodes*OutputNodes + i] = setWeight(LearningRate * OutputDelta[i] + Momentum * getWeight(ChangeOutputWeights[HiddenNodes*OutputNodes + i]));
        OutputWeights[HiddenNodes*OutputNodes + i] += ChangeOutputWeights[HiddenNodes*OutputNodes + i];
        for(int j = 0 ; j < HiddenNodes ; j++ ) {
            ChangeOutputWeights[j*OutputNodes + i] = setWeight(LearningRate * Hidden[j] * OutputDelta[i] + Momentum * getWeight(ChangeOutputWeights[j*OutputNodes + i]));
            OutputWeights[j*OutputNodes + i] += ChangeOutputWeights[j*OutputNodes + i];
        }
    }

    return error;
}


float* NeuralNetwork::get_output(){
    return Output;
}

weightType* NeuralNetwork::get_HiddenWeights(){
    return HiddenWeights;
}

weightType* NeuralNetwork::get_OutputWeights(){
    return OutputWeights;
}

float NeuralNetwork::get_error(){
    return Error;
}
