#include "../include/Transformer.h"

Matrix SelfAttention::compute_attention(Matrix Q, Matrix K, Matrix V) {
    // 1. اضرب Q في K مقلوبة (Q * K^T)
    Matrix K_T = K.transpose();
    Matrix scores = Q.multiply(K_T);
    
    // 2. اقسم على الجذر التربيعي لحجم الـ Key (عشان الأرقام متكبرش أوي)
    double d_k = K.cols;
    Matrix scaled_scores = scores.scalar_multiply(1.0 / sqrt(d_k));
    
    // 3. طبق الـ Softmax عشان تطلع نسب الانتباه
    Matrix attention_weights = scaled_scores.softmax();
    
    // 4. اضرب النتيجة في V (القيمة)
    Matrix output = attention_weights.multiply(V);
    
    return output;
}