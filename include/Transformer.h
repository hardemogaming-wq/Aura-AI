#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include "Matrix.h"
#include "LayerNorm.h"
#include "FeedForward.h"
#include <cmath>

class SelfAttention {
public:
    // دالة الانتباه الذاتي
    Matrix compute_attention(Matrix Q, Matrix K, Matrix V);
};

class TransformerBlock {
public:
    SelfAttention attention;
    LayerNorm norm1;
    LayerNorm norm2;
    FeedForward ffn;

    // Constructor بياخد أبعاد الكلمات (d_model) وحجم التوسيع في الـ FFN
    TransformerBlock(int d_model, int d_ff) : ffn(d_model, d_ff) {}

    Matrix forward(Matrix input) {
        // 1. الانتباه الذاتي (Self-Attention)
        // في النسخة المبسطة دي، هندخل نفس الـ input كـ Query و Key و Value
        Matrix attn_output = attention.compute_attention(input, input, input);

        // 2. الوصلة المتبقية الأولى (Add) + التطبيع (Norm)
        Matrix add1(input.rows, input.cols);
        for(int i = 0; i < input.rows; i++) {
            for(int j = 0; j < input.cols; j++) {
                add1.data[i][j] = input.data[i][j] + attn_output.data[i][j];
            }
        }
        norm1.forward(add1); // الدالة دي بتعدل على add1 مباشرة زي ما برمجناها

        // 3. التمرير الأمامي (Feed-Forward)
        Matrix ff_output = ffn.forward(add1);

        // 4. الوصلة المتبقية الثانية (Add) + التطبيع (Norm)
        Matrix add2(add1.rows, add1.cols);
        for(int i = 0; i < add1.rows; i++) {
            for(int j = 0; j < add1.cols; j++) {
                add2.data[i][j] = add1.data[i][j] + ff_output.data[i][j];
            }
        }
        norm2.forward(add2);

        return add2; // دي النتيجة النهائية للـ Block!
    }
};

#endif