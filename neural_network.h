#ifndef NEURAL_NETWORK
#define NEURAL_NETWORK


/******************************************************************
 * Network Configuration - customized per network 
 ******************************************************************/

static const int PatternCount = 3;
static const int InputNodes = 650;
static const int HiddenNodes = 5;
static const int OutputNodes = 3;
static const float InitialWeightMax = 0.05;
static const float InitialWeightMin = -0.05;

using weightType = int8_t; // int8_t / int16_t, float
static const float weightFactor = 100.f;

static const int hiddenWeightsAmt = (InputNodes + 1) * HiddenNodes;
static const int outputWeightsAmt = (HiddenNodes + 1) * OutputNodes;

class NeuralNetwork {
    public:

        void initialize(float LearningRate, float Momentum, int DropoutRate);
        // ~NeuralNetwork();

        // void initWeights();
        float forward(const float Input[], const float Target[]);
        float backward(float Input[], const float Target[]); // Input will be changed!!

        float* get_output();

        weightType* get_HiddenWeights();
        weightType* get_OutputWeights();

        float get_error();

        float getWeight(weightType val);
        weightType setWeight(float val);

        
        
    private:

        float Hidden[HiddenNodes] = {};
        float Output[OutputNodes] = {};
        weightType HiddenWeights[hiddenWeightsAmt] = {};
        weightType OutputWeights[outputWeightsAmt] = {};
        float HiddenDelta[HiddenNodes] = {};
        float OutputDelta[OutputNodes] = {};
        weightType ChangeHiddenWeights[hiddenWeightsAmt] = {};
        weightType ChangeOutputWeights[outputWeightsAmt] = {};

        float Error;
        float LearningRate = 0.6;
        float Momentum = 0.9;
        // From 0 to 100, the percentage of input features that will be set to 0
        int DropoutRate = 0;
};


#endif
