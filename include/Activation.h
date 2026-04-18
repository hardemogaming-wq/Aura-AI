#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "Matrix.h"

class Activation {
public:
    // Sigmoid activation function: 1 / (1 + e^-x)
    static double sigmoid(double x);
    
    // Sigmoid derivative: σ(x) * (1 - σ(x))
    static double sigmoid_derivative(double sigma_x);
    
    // ReLU activation function: max(0, x)
    static double relu(double x);
    
    // ReLU derivative: 1 if x > 0, else 0
    static double relu_derivative(double x);
    
    // Tanh activation function: (e^x - e^-x) / (e^x + e^-x)
    static double tanh_activation(double x);
    
    // Tanh derivative: 1 - tanh(x)^2
    static double tanh_derivative(double tanh_x);
};

#endif
