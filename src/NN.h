#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

typedef unsigned int uint;

template <uint HL_SIZE>
class NeuralNetwork {
    public:
        static const uint InputNodes = 650;
        static const uint OutputNodes = 4;
        
        const float InitialWeightMax = 0.05;
        const float InitialWeightMin = -0.05;

        uint getHiddenWeightsAmt() {
            return(InputNodes + 1) * HL_SIZE;
        }

        uint getOutputWeightsAmt() {
            return (HL_SIZE + 1) * OutputNodes;
        }

        static float sigmoid(float x) {
            return 1.0 / (1.0 + exp(-x));
        }

        void initialize(float learningRate, float momentum) {
            this->LearningRate = learningRate;
            this->Momentum = momentum;
            this->activation = NeuralNetwork<HL_SIZE>::sigmoid;

            for (uint i = 0; i < getHiddenWeightsAmt(); ++i) {
                HiddenWeights[i] = random(InitialWeightMin*100, InitialWeightMax*100)/100.f; // Random generates ints
            }

            for (uint i = 0; i < getOutputWeightsAmt(); ++i) {
                OutputWeights[i] = random(InitialWeightMin*100, InitialWeightMax*100)/100.f; // Random generates ints
            }
        }

        float forward(const float Input[], const float Target[]) {
            float error = 0;

            // Compute hidden layer activations
            for (uint i = 0; i < HL_SIZE; i++) {
                float Accum = HiddenWeights[InputNodes * HL_SIZE + i];
                for (uint j = 0; j < InputNodes; j++) {
                    Accum += Input[j] * HiddenWeights[j * HL_SIZE + i];
                }
                Hidden[i] = this->activation(Accum);
            }
            
            // Compute output layer activations and calculate errors
            for (uint i = 0; i < OutputNodes; i++) {
                float Accum = OutputWeights[HL_SIZE * OutputNodes + i];
                for (uint j = 0; j < HL_SIZE; j++) {
                    Accum += Hidden[j] * OutputWeights[j*OutputNodes + i];
                }
                Output[i] = this->activation(Accum);
                error += (1.0/OutputNodes) * (Target[i] - Output[i]) * (Target[i] - Output[i]);
            }
            
            return error;
        }

        float backward(const float Input[], const float Target[]){
            float error = 0;

            // Forward
            // Compute hidden layer activations
            for (uint i = 0; i < HL_SIZE; i++) {
                float Accum = HiddenWeights[InputNodes*HL_SIZE + i];
                for (uint j = 0; j < InputNodes; j++) {
                    Accum += Input[j] * HiddenWeights[j*HL_SIZE + i];
                }
                Hidden[i] = this->activation(Accum);
            }

            // Compute output layer activations and calculate errors
            for (uint i = 0; i < OutputNodes; i++) {
                float Accum = OutputWeights[HL_SIZE*OutputNodes + i];
                for (uint j = 0; j < HL_SIZE; j++) {
                    Accum += Hidden[j] * OutputWeights[j*OutputNodes + i];
                }
                Output[i] = this->activation(Accum);
                OutputDelta[i] = (Target[i] - Output[i]) * Output[i] * (1.0 - Output[i]);
                error += (1.0/OutputNodes) * (Target[i] - Output[i]) * (Target[i] - Output[i]);
            }
            // End forward

            // Backward
            // Backpropagate errors to hidden layer
            for(uint i = 0 ; i < HL_SIZE ; i++ ) {    
                float Accum = 0.0 ;
                for(uint j = 0 ; j < OutputNodes ; j++ ) {
                    Accum += OutputWeights[i*OutputNodes + j] * OutputDelta[j] ;
                }
                HiddenDelta[i] = Accum * Hidden[i] * (1.0 - Hidden[i]) ;
            }

            // Update Inner-->Hidden Weights
            for(uint i = 0 ; i < HL_SIZE ; i++ ) {     
                ChangeHiddenWeights[InputNodes*HL_SIZE + i] = LearningRate * HiddenDelta[i] + Momentum * ChangeHiddenWeights[InputNodes*HL_SIZE + i] ;
                HiddenWeights[InputNodes*HL_SIZE + i] += ChangeHiddenWeights[InputNodes*HL_SIZE + i] ;
                for(uint j = 0 ; j < InputNodes ; j++ ) { 
                    ChangeHiddenWeights[j*HL_SIZE + i] = LearningRate * Input[j] * HiddenDelta[i] + Momentum * ChangeHiddenWeights[j*HL_SIZE + i];
                    HiddenWeights[j*HL_SIZE + i] += ChangeHiddenWeights[j*HL_SIZE + i] ;
                }
            }

            // Update Hidden-->Output Weights
            for(uint i = 0 ; i < OutputNodes ; i ++ ) {    
                ChangeOutputWeights[HL_SIZE*OutputNodes + i] = LearningRate * OutputDelta[i] + Momentum * ChangeOutputWeights[HL_SIZE*OutputNodes + i] ;
                OutputWeights[HL_SIZE*OutputNodes + i] += ChangeOutputWeights[HL_SIZE*OutputNodes + i] ;
                for(uint j = 0 ; j < HL_SIZE ; j++ ) {
                    ChangeOutputWeights[j*OutputNodes + i] = LearningRate * Hidden[j] * OutputDelta[i] + Momentum * ChangeOutputWeights[j*OutputNodes + i] ;
                    OutputWeights[j*OutputNodes + i] += ChangeOutputWeights[j*OutputNodes + i] ;
                }
            }
            return error;
        }
        float* get_output() {
            return Output;
        }
        float* getHiddenWeights() {
            return HiddenWeights;
        }
        float* getOutputWeights() {
            return OutputWeights;
        }
        float get_error(){
            return Error;
        }

    private:
        float Hidden[HL_SIZE] = {};
        float Output[OutputNodes] = {};
        float HiddenWeights[(InputNodes+1) * HL_SIZE] = {};
        float OutputWeights[(HL_SIZE+1) * OutputNodes] = {};
        float HiddenDelta[HL_SIZE] = {};
        float OutputDelta[OutputNodes] = {};
        float ChangeHiddenWeights[(InputNodes+1) * HL_SIZE] = {};
        float ChangeOutputWeights[(HL_SIZE+1) * OutputNodes] = {};
        float (*activation)(float);
        float Error;
        float LearningRate;
        float Momentum;
};


#endif /* NEURAL_NETWORK_H */