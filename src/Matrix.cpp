#include "Matrix.h"
#include <iostream>
#include <stdexcept>
#include <cmath>

Matrix::Matrix(int r, int c) : rows(r), cols(c), data(r, std::vector<elem_t>(c, 0)) {}

Matrix Matrix::multiply(const Matrix& b) const {
    if (cols != b.rows) throw std::invalid_argument("Matrix dimensions mismatch");
    Matrix c(rows, b.cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < b.cols; ++j) {
            elem_t sum = 0;
            for (int k = 0; k < cols; ++k) {
                sum += data[i][k] * b.data[k][j];
            }
            c.data[i][j] = sum;
        }
    }
    return c;
}

Matrix Matrix::add(const Matrix& b) const {
    if (rows != b.rows || cols != b.cols) throw std::invalid_argument("Matrix dimensions mismatch");
    Matrix c(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            c.data[i][j] = data[i][j] + b.data[i][j];
        }
    }
    return c;
}

Matrix Matrix::subtract(const Matrix& b) const {
    if (rows != b.rows || cols != b.cols) throw std::invalid_argument("Matrix dimensions mismatch");
    Matrix c(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            c.data[i][j] = data[i][j] - b.data[i][j];
        }
    }
    return c;
}

Matrix Matrix::relu() const {
    Matrix c(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            c.data[i][j] = data[i][j] > 0 ? data[i][j] : 0;
        }
    }
    return c;
}

Matrix Matrix::sigmoid() const {
    Matrix c(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            elem_t val = data[i][j];
            c.data[i][j] = 1.0 / (1.0 + exp(-val));
        }
    }
    return c;
}

Matrix Matrix::transpose() const {
    Matrix t(cols, rows);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            t.data[j][i] = data[i][j];
        }
    }
    return t;
}

Matrix Matrix::scalar_multiply(double scalar) const {
    Matrix c(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            c.data[i][j] = data[i][j] * scalar;
        }
    }
    return c;
}

Matrix Matrix::multiply_elementwise(const Matrix& b) const {
    if (rows != b.rows || cols != b.cols) throw std::invalid_argument("Matrix dimensions mismatch for element-wise multiply");
    Matrix c(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            c.data[i][j] = data[i][j] * b.data[i][j];
        }
    }
    return c;
}

Matrix::QuantizedMatrix Matrix::quantize_int8() const {
    QuantizedMatrix q;
    q.rows = rows;
    q.cols = cols;
    q.data.assign(rows * cols, 0);

    double max_abs = 0.0;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            max_abs = std::max(max_abs, std::abs(data[i][j]));
        }
    }

    q.scale = max_abs > 0.0 ? 127.0 / max_abs : 1.0;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int idx = i * cols + j;
            q.data[idx] = static_cast<int8_t>(std::round(data[i][j] * q.scale));
        }
    }

    return q;
}

Matrix Matrix::dequantize_from_int8(const QuantizedMatrix& q) const {
    Matrix m(q.rows, q.cols);
    double inv_scale = q.scale != 0.0 ? 1.0 / q.scale : 1.0;
    for (int i = 0; i < q.rows; ++i) {
        for (int j = 0; j < q.cols; ++j) {
            int idx = i * q.cols + j;
            m.data[i][j] = q.data[idx] * inv_scale;
        }
    }
    return m;
}

std::vector<int8_t> Matrix::quantize_int8(double& scale) const {
    elem_t max_abs = 0.0;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            max_abs = std::max(max_abs, std::abs(data[i][j]));
        }
    }
    if (max_abs < 1e-12) {
        max_abs = 1.0;
    }

    scale = 127.0 / max_abs;
    std::vector<int8_t> quantized(rows * cols);
    int idx = 0;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int q = static_cast<int>(std::lround(data[i][j] * scale));
            if (q > 127) q = 127;
            if (q < -128) q = -128;
            quantized[idx++] = static_cast<int8_t>(q);
        }
    }
    return quantized;
}

Matrix Matrix::dequantize_int8(const std::vector<int8_t>& data8, int rows, int cols, double scale) {
    if (static_cast<int>(data8.size()) != rows * cols) {
        throw std::invalid_argument("Quantized data size does not match matrix dimensions");
    }

    Matrix result(rows, cols);
    int idx = 0;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result.data[i][j] = static_cast<double>(data8[idx++]) / scale;
        }
    }
    return result;
}

void Matrix::print() const {
    for (const auto& row : data) {
        for (elem_t val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
}

// تحويل القيم لاحتمالات مجموعها 1
Matrix Matrix::softmax() const {
    Matrix result(rows, cols);
    double sum = 0.0;
    
    // أولاً: نحسب الـ exp لكل رقم ونجمعهم
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result.data[i][j] = exp(data[i][j]);
            sum += result.data[i][j];
        }
    }
    
    // ثانياً: نقسم كل رقم على المجموع الكلي
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result.data[i][j] /= sum;
        }
    }
    return result;
}