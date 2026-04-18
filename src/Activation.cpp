#include "Activation.h"
#include <cmath>

// Sigmoid activation function
// معادلة السيجمويد: σ(x) = 1 / (1 + e^-x)
// تحول القيم إلى نطاق بين 0 و 1
double Activation::sigmoid(double x) {
    // لتجنب Overflow في e^-x للقيم الكبيرة جداً
    if (x > 20) return 1.0;
    if (x < -20) return 0.0;
    return 1.0 / (1.0 + std::exp(-x));
}

// مشتقة السيجمويد
// إذا كانت σ(x) هي القيمة بعد تمرير sigmoid، فمشتقتها = σ(x) * (1 - σ(x))
// هذا يُستخدم في Backpropagation للتعليم
double Activation::sigmoid_derivative(double sigma_x) {
    return sigma_x * (1.0 - sigma_x);
}

// ReLU activation function (Rectified Linear Unit)
// معادلة ReLU: f(x) = max(0, x)
// تحافظ على القيم الموجبة وتحول السالبة إلى 0
// ميزتها أنها أسرع وتقلل مشاكل التشبع (Vanishing Gradient)
double Activation::relu(double x) {
    return x > 0.0 ? x : 0.0;
}

// مشتقة ReLU
// تكون 1 عندما x > 0، وتكون 0 عندما x <= 0
double Activation::relu_derivative(double x) {
    return x > 0.0 ? 1.0 : 0.0;
}

// Tanh activation function (Hyperbolic Tangent)
// معادلة Tanh: tanh(x) = (e^x - e^-x) / (e^x + e^-x)
// تحول القيم إلى نطاق بين -1 و 1
// أفضل من sigmoid في بعض الحالات
double Activation::tanh_activation(double x) {
    return std::tanh(x);
}

// مشتقة Tanh
// إذا كانت tanh_x هي القيمة بعد تمرير tanh، فمشتقتها = 1 - tanh(x)^2
double Activation::tanh_derivative(double tanh_x) {
    return 1.0 - (tanh_x * tanh_x);
}
