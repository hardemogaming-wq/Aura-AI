#include "NeuralNetwork.h"
#include <stdexcept>
#include <iostream>
#include "Loss.h"
#include "Activation.h"
#include <fstream>

NeuralNetwork::NeuralNetwork(const std::vector<int>& layer_sizes, 
                             const std::vector<std::string>& activations,
                             double lr) 
    : learning_rate(lr) {
    
    // التحقق من أن عدد دوال التنشيط تطابق عدد الطبقات
    if (activations.size() != layer_sizes.size() - 1) {
        throw std::invalid_argument("Activation functions must match number of layers");
    }
    
    // إنشاء الطبقات
    // مثال: layer_sizes = [784, 128, 64, 10]
    //       يعني: 784 input -> 128 hidden -> 64 hidden -> 10 output
    
    for (size_t i = 0; i < layer_sizes.size() - 1; ++i) {
        int input_size = layer_sizes[i];
        int output_size = layer_sizes[i + 1];
        std::string activation = activations[i];
        
        // تحقق أن دالة التنشيط معروفة
        if (activation != "relu" && activation != "sigmoid" && 
            activation != "tanh" && activation != "none") {
            throw std::invalid_argument("Unknown activation: " + activation);
        }
        
        layers.emplace_back(input_size, output_size, activation);
    }
}

// Forward pass: مرر البيانات عبر جميع الطبقات
Matrix NeuralNetwork::forward(const Matrix& input) {
    
    // تحقق أن المدخل هو column vector
    if (input.cols != 1) {
        throw std::invalid_argument("Input must be a column vector (cols=1)");
    }
    
    // تحقق أن حجم المدخل يطابق حجم الطبقة الأولى
    if (input.rows != layers[0].getInputSize()) {
        throw std::invalid_argument("Input size mismatch with network");
    }
    
    layer_inputs.clear();
    layer_outputs.clear();
    
    Matrix current_output = input;
    layer_inputs.push_back(current_output);
    
    // مرر البيانات عبر كل طبقة
    for (size_t i = 0; i < layers.size(); ++i) {
        current_output = layers[i].forward(current_output);
        layer_outputs.push_back(current_output);
        if (i < layers.size() - 1) {
            layer_inputs.push_back(current_output);
        }
    }
    
    return current_output;
}

// Backward pass: تعلم من الخطأ
void NeuralNetwork::backward(const Matrix& input, const Matrix& target, double learning_rate) {
    // احسب مشتقة الخطأ للمخرجات
    Matrix error = Loss::mse_derivative(layer_outputs.back(), target);
    
    // ابدأ من الطبقة الأخيرة
    for (int i = layers.size() - 1; i >= 0; --i) {
        Layer& layer = layers[i];
        
        // حساب مشتقة دالة التنشيط
        Matrix activation_deriv(layer_outputs[i].rows, layer_outputs[i].cols);
        for (int r = 0; r < layer_outputs[i].rows; ++r) {
            for (int c = 0; c < layer_outputs[i].cols; ++c) {
                double val = layer_outputs[i].data[r][c];
                if (layer.getActivation() == "sigmoid") {
                    activation_deriv.data[r][c] = Activation::sigmoid_derivative(val);
                } else if (layer.getActivation() == "relu") {
                    activation_deriv.data[r][c] = Activation::relu_derivative(val);
                } else if (layer.getActivation() == "tanh") {
                    activation_deriv.data[r][c] = Activation::tanh_derivative(val);
                } else {
                    activation_deriv.data[r][c] = 1.0; // none
                }
            }
        }
        
        // ضرب إشارة الخطأ في مشتقة التنشيط
        error = error.multiply_elementwise(activation_deriv);
        
        // حساب التدرج للأوزان
        Matrix grad_weights = error.multiply(layer_inputs[i].transpose());
        
        // تحديث الأوزان
        layer.setWeights(layer.getWeights().subtract(grad_weights.scalar_multiply(learning_rate)));
        
        // تحديث الـ bias
        layer.setBias(layer.getBias().subtract(error.scalar_multiply(learning_rate)));
        
        // تمرير الخطأ للطبقة السابقة
        if (i > 0) {
            error = layer.getWeights().transpose().multiply(error);
        }
    }
}

// Save model to file
void NeuralNetwork::save_model(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }
    
    // Write number of layers
    file << layers.size() << std::endl;
    
    for (const auto& layer : layers) {
        // Write weights
        file << layer.getWeights().rows << " " << layer.getWeights().cols << std::endl;
        for (int i = 0; i < layer.getWeights().rows; ++i) {
            for (int j = 0; j < layer.getWeights().cols; ++j) {
                file << layer.getWeights().data[i][j] << " ";
            }
            file << std::endl;
        }
        
        // Write bias
        file << layer.getBias().rows << " " << layer.getBias().cols << std::endl;
        for (int i = 0; i < layer.getBias().rows; ++i) {
            for (int j = 0; j < layer.getBias().cols; ++j) {
                file << layer.getBias().data[i][j] << " ";
            }
            file << std::endl;
        }
        
        // Write activation
        file << layer.getActivation() << std::endl;
    }
    
    file.close();
}

// Load model from file
void NeuralNetwork::load_model(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for reading: " + filename);
    }
    
    int num_layers;
    file >> num_layers;
    
    layers.clear();
    for (int l = 0; l < num_layers; ++l) {
        int w_rows, w_cols;
        file >> w_rows >> w_cols;
        Matrix weights(w_rows, w_cols);
        for (int i = 0; i < w_rows; ++i) {
            for (int j = 0; j < w_cols; ++j) {
                file >> weights.data[i][j];
            }
        }
        
        int b_rows, b_cols;
        file >> b_rows >> b_cols;
        Matrix bias(b_rows, b_cols);
        for (int i = 0; i < b_rows; ++i) {
            for (int j = 0; j < b_cols; ++j) {
                file >> bias.data[i][j];
            }
        }
        
        std::string activation;
        file >> activation;
        
        Layer layer(1, 1, activation); // dummy sizes, will set manually
        layer.setWeights(weights);
        layer.setBias(bias);
        layers.push_back(layer);
    }
    
    file.close();
}

std::vector<Matrix::QuantizedMatrix> NeuralNetwork::quantize_weights() const {
    std::vector<Matrix::QuantizedMatrix> quantized;
    quantized.reserve(layers.size() * 2);

    for (const auto& layer : layers) {
        quantized.push_back(layer.getWeights().quantize_int8());
        quantized.push_back(layer.getBias().quantize_int8());
    }

    return quantized;
}

void NeuralNetwork::save_quantized_model(const std::string& filename) const {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }

    int layer_count = static_cast<int>(layers.size());
    file.write(reinterpret_cast<const char*>(&layer_count), sizeof(layer_count));

    for (const auto& layer : layers) {
        const std::string& activation = layer.getActivation();
        int activation_size = static_cast<int>(activation.size());
        file.write(reinterpret_cast<const char*>(&activation_size), sizeof(activation_size));
        file.write(activation.data(), activation_size);

        auto weight_q = layer.getWeights().quantize_int8();
        int w_rows = weight_q.rows;
        int w_cols = weight_q.cols;
        file.write(reinterpret_cast<const char*>(&w_rows), sizeof(w_rows));
        file.write(reinterpret_cast<const char*>(&w_cols), sizeof(w_cols));
        file.write(reinterpret_cast<const char*>(&weight_q.scale), sizeof(weight_q.scale));
        if (!weight_q.data.empty()) {
            file.write(reinterpret_cast<const char*>(weight_q.data.data()), w_rows * w_cols * sizeof(int8_t));
        }

        auto bias_q = layer.getBias().quantize_int8();
        int b_rows = bias_q.rows;
        int b_cols = bias_q.cols;
        file.write(reinterpret_cast<const char*>(&b_rows), sizeof(b_rows));
        file.write(reinterpret_cast<const char*>(&b_cols), sizeof(b_cols));
        file.write(reinterpret_cast<const char*>(&bias_q.scale), sizeof(bias_q.scale));
        if (!bias_q.data.empty()) {
            file.write(reinterpret_cast<const char*>(bias_q.data.data()), b_rows * b_cols * sizeof(int8_t));
        }
    }

    file.close();
}

void NeuralNetwork::load_quantized_model(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for reading: " + filename);
    }

    int layer_count;
    file.read(reinterpret_cast<char*>(&layer_count), sizeof(layer_count));
    if (!file) {
        throw std::runtime_error("Invalid quantized model file: " + filename);
    }

    layers.clear();
    for (int l = 0; l < layer_count; ++l) {
        int activation_size;
        file.read(reinterpret_cast<char*>(&activation_size), sizeof(activation_size));
        if (activation_size <= 0) {
            throw std::runtime_error("Invalid activation string size in quantized model");
        }

        std::string activation(activation_size, '\0');
        file.read(&activation[0], activation_size);

        int w_rows, w_cols;
        double w_scale;
        file.read(reinterpret_cast<char*>(&w_rows), sizeof(w_rows));
        file.read(reinterpret_cast<char*>(&w_cols), sizeof(w_cols));
        file.read(reinterpret_cast<char*>(&w_scale), sizeof(w_scale));
        if (w_rows <= 0 || w_cols <= 0) {
            throw std::runtime_error("Invalid quantized weight matrix dimensions");
        }
        std::vector<int8_t> w_data(w_rows * w_cols);
        file.read(reinterpret_cast<char*>(w_data.data()), w_data.size() * sizeof(int8_t));
        Matrix weights = Matrix::dequantize_int8(w_data, w_rows, w_cols, w_scale);

        int b_rows, b_cols;
        double b_scale;
        file.read(reinterpret_cast<char*>(&b_rows), sizeof(b_rows));
        file.read(reinterpret_cast<char*>(&b_cols), sizeof(b_cols));
        file.read(reinterpret_cast<char*>(&b_scale), sizeof(b_scale));
        if (b_rows <= 0 || b_cols <= 0) {
            throw std::runtime_error("Invalid quantized bias matrix dimensions");
        }
        std::vector<int8_t> b_data(b_rows * b_cols);
        file.read(reinterpret_cast<char*>(b_data.data()), b_data.size() * sizeof(int8_t));
        Matrix bias = Matrix::dequantize_int8(b_data, b_rows, b_cols, b_scale);

        Layer layer(w_cols, w_rows, activation);
        layer.setWeights(weights);
        layer.setBias(bias);
        layers.push_back(layer);
    }

    file.close();
}
