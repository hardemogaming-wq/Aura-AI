#ifndef LAYERNORM_H
#define LAYERNORM_H

#include "Matrix.h"
#include <cmath>
#include <vector>

class LayerNorm {
public:
    double epsilon = 1e-5;

    // الدالة دي بتعمل Normalization لكل صف (كلمة) لوحده
    void forward(Matrix& input) {
        for (int i = 0; i < input.rows; i++) {
            double mean = 0.0;
            double variance = 0.0;

            // 1. حساب المتوسط (Mean) للصف
            for (int j = 0; j < input.cols; j++) {
                mean += input.data[i][j];
            }
            mean /= input.cols;

            // 2. حساب التباين (Variance)
            for (int j = 0; j < input.cols; j++) {
                variance += pow(input.data[i][j] - mean, 2);
            }
            variance /= input.cols;

            // 3. تعديل القيم (Normalize)
            for (int j = 0; j < input.cols; j++) {
                input.data[i][j] = (input.data[i][j] - mean) / sqrt(variance + epsilon);
            }
        }
    }
};

#endif