#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <cstdint>

using elem_t = double;
using acc_t = double;

class Matrix {
public:
    struct QuantizedMatrix {
        int rows;
        int cols;
        double scale;
        std::vector<int8_t> data;
    };

    int rows, cols;
    std::vector<std::vector<elem_t>> data;

    Matrix(int r, int c);
    Matrix multiply(const Matrix& b) const;
    Matrix add(const Matrix& b) const;
    Matrix subtract(const Matrix& b) const;
    Matrix relu() const;
    Matrix sigmoid() const;
    // تحويل القيم لاحتمالات مجموعها 1
    Matrix softmax() const;
    Matrix transpose() const;
    Matrix scalar_multiply(double scalar) const;
    Matrix multiply_elementwise(const Matrix& b) const;
    QuantizedMatrix quantize_int8() const;
    Matrix dequantize_from_int8(const QuantizedMatrix& q) const;
    std::vector<int8_t> quantize_int8(double& scale) const;
    static Matrix dequantize_int8(const std::vector<int8_t>& data8, int rows, int cols, double scale);
    void print() const;
};

#endif