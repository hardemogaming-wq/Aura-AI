#include "Loss.h"
#include <stdexcept>

// دالة بتحسب نسبة الخطأ الكلية عشان نطبعها ونشوف الشبكة بتتحسن ولا لأ
double Loss::mse_loss(const Matrix& predictions, const Matrix& targets) {
    if (predictions.rows != targets.rows || predictions.cols != targets.cols) {
        throw std::invalid_argument("Predictions and targets must have the same dimensions");
    }
    double loss = 0.0;
    for (int i = 0; i < predictions.rows; i++) {
        for (int j = 0; j < predictions.cols; j++) {
            double diff = predictions.data[i][j] - targets.data[i][j];
            loss += diff * diff;
        }
    }
    return loss / (predictions.rows * predictions.cols);
}

// الدالة دي بترجع مصفوفة فيها نسبة التعديل اللي محتاجين نعمله
Matrix Loss::mse_derivative(const Matrix& predictions, const Matrix& targets) {
    if (predictions.rows != targets.rows || predictions.cols != targets.cols) {
        throw std::invalid_argument("Predictions and targets must have the same dimensions");
    }
    Matrix derivative(predictions.rows, predictions.cols);
    for (int i = 0; i < predictions.rows; i++) {
        for (int j = 0; j < predictions.cols; j++) {
            // تفاضل الـ MSE
            derivative.data[i][j] = 2.0 * (predictions.data[i][j] - targets.data[i][j]) / (predictions.rows * predictions.cols);
        }
    }
    return derivative;
}