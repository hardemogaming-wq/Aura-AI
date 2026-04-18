#ifndef LOSS_H
#define LOSS_H

#include "Matrix.h"

class Loss {
public:
    // حساب Mean Squared Error
    static double mse_loss(const Matrix& predictions, const Matrix& targets);
    
    // مشتقة MSE للـ Backpropagation
    static Matrix mse_derivative(const Matrix& predictions, const Matrix& targets);
};

#endif