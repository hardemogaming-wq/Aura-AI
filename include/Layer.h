#ifndef LAYER_H
#define LAYER_H

#include "Matrix.h"
#include <string>

class Layer {
private:
    Matrix weights;
    Matrix bias;
    std::string activation;

public:
    Layer(int input_size, int output_size, std::string act);
    Matrix forward(const Matrix& input);
    
    // Getter methods
    const Matrix& getWeights() const { return weights; }
    const Matrix& getBias() const { return bias; }
    int getInputSize() const { return weights.cols; }
    int getOutputSize() const { return weights.rows; }
    std::string getActivation() const { return activation; }
    
    // Setter methods for training
    void setWeights(const Matrix& w) { weights = w; }
    void setBias(const Matrix& b) { bias = b; }
};

#endif