#ifndef EMBEDDING_H
#define EMBEDDING_H

#include "Matrix.h"
#include <cstdlib>

class Embedding {
public:
    Matrix weights; // مصفوفة الأوزان اللي شايلة معاني الكلمات

    // Constructor: بياخد حجم القاموس، وحجم الـ Vector
    Embedding(int vocab_size, int embedding_dim) : weights(vocab_size, embedding_dim) {
        // تهيئة الأوزان بأرقام عشوائية صغيرة
        for (int i = 0; i < vocab_size; i++) {
            for (int j = 0; j < embedding_dim; j++) {
                weights.data[i][j] = ((double) rand() / (RAND_MAX)) * 0.1;
            }
        }
    }

    // دالة بتاخد رقم الكلمة (ID) وترجع الـ Matrix بتاعتها
    Matrix get_embedding(int token_id) {
        // بنعمل مصفوفة (1 × حجم الـ Embedding)
        Matrix word_vector(1, weights.cols);
        
        for (int j = 0; j < weights.cols; j++) {
            word_vector.data[0][j] = weights.data[token_id][j];
        }
        
        return word_vector;
    }
};

#endif