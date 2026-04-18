#include "Layer.h"
#include <stdexcept>
#include <cstdlib>
#include <ctime>

Layer::Layer(int input_size, int output_size, std::string act) 
    : weights(output_size, input_size), bias(output_size, 1), activation(act) {
    // Initialize weights and bias with small random values
    std::srand(std::time(nullptr));
    for (int i = 0; i < output_size; ++i) {
        for (int j = 0; j < input_size; ++j) {
            weights.data[i][j] = (std::rand() % 200 - 100) / 100.0; // random between -1 and 1
        }
        bias.data[i][0] = (std::rand() % 200 - 100) / 100.0;
    }
}

Matrix Layer::forward(const Matrix& input) {
    // Assume input is column vector, rows = input_size, cols=1
    if (input.cols != 1 || input.rows != weights.cols) {
        throw std::invalid_argument("Input dimensions mismatch");
    }
    Matrix out = weights.multiply(input).add(bias);
    if (activation == "relu") {
        out = out.relu();
    } else if (activation == "sigmoid") {
        out = out.sigmoid();
    }
    // else none
    return out;
}