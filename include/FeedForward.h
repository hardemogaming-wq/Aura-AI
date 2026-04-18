#ifndef FEEDFORWARD_H
#define FEEDFORWARD_H

#include "Matrix.h"
#include "Layer.h" // سنستخدم كلاس Layer اللي عملناه في الأول

class FeedForward {
public:
    Layer layer1; // الطبقة الأولى (توسيع)
    Layer layer2; // الطبقة الثانية (تقليص)

    FeedForward(int d_model, int d_ff) 
        : layer1(d_model, d_ff, "relu"), layer2(d_ff, d_model, "none") {}

    Matrix forward(const Matrix& input) {
        // معالجة كل صف (كلمة) على حدة
        Matrix output(input.rows, input.cols);
        for (int i = 0; i < input.rows; i++) {
            // استخراج الصف كـ vector عمودي
            Matrix row_vector(input.cols, 1);
            for (int j = 0; j < input.cols; j++) {
                row_vector.data[j][0] = input.data[i][j];
            }
            
            // تمرير عبر الطبقتين
            Matrix x = layer1.forward(row_vector);
            Matrix result = layer2.forward(x);
            
            // وضع النتيجة في الصف المقابل
            for (int j = 0; j < input.cols; j++) {
                output.data[i][j] = result.data[j][0];
            }
        }
        return output;
    }
};

#endif