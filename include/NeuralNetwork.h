#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "Layer.h"
#include <vector>
#include <string>

class NeuralNetwork {
private:
    std::vector<Layer> layers;
    double learning_rate;
    std::vector<Matrix> layer_inputs;
    std::vector<Matrix> layer_outputs;
    
public:
    NeuralNetwork() = default;

    // Constructor: إنشاء شبكة عصبية جديدة
    // input_sizes: عدد المدخلات والمخرجات لكل طبقة
    // activations: دالة التنشيط لكل طبقة
    NeuralNetwork(const std::vector<int>& layer_sizes, 
                  const std::vector<std::string>& activations,
                  double lr = 0.01);
    
    // Forward pass: مرر البيانات عبر الشبكة
    // input: مصفوفة المدخلات (عادة column vector)
    // العودة: مصفوفة المخرجات من الطبقة الأخيرة
    Matrix forward(const Matrix& input);

    // Backward pass: تعلم من الخطأ
    void backward(const Matrix& input, const Matrix& target, double learning_rate);
    
    // Save model to file
    void save_model(const std::string& filename) const;
    
    // Load model from file
    void load_model(const std::string& filename);
    
    // Load a quantized model from binary file and reconstruct layers
    void load_quantized_model(const std::string& filename);

    // Compress weights to int8 representation
    std::vector<Matrix::QuantizedMatrix> quantize_weights() const;

    // Save quantized weights and bias into a binary file
    void save_quantized_model(const std::string& filename) const;
    
    // حصول على عدد الطبقات
    int getNumLayers() const { return layers.size(); }
    
    // حصول على طبقة محددة
    Layer& getLayer(int index) { return layers[index]; }
    
    // تعيين معدل التعليم
    void setLearningRate(double lr) { learning_rate = lr; }
};

#endif
