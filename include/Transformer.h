#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include "Matrix.h"
#include <cmath>

class SelfAttention {
public:
    // دالة الانتباه الذاتي
    Matrix compute_attention(Matrix Q, Matrix K, Matrix V);
};

#endif