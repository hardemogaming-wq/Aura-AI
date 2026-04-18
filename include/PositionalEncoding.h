#ifndef POSITIONALENCODING_H
#define POSITIONALENCODING_H

#include "Matrix.h"
#include <cmath>

class PositionalEncoding {
public:
    // الدالة دي بتاخد مصفوفة الجملة (كل صف فيها عبارة عن كلمة)
    static void add_positional_encoding(Matrix& seq_embeddings) {
        int seq_length = seq_embeddings.rows; // عدد الكلمات في الجملة
        int embed_dim = seq_embeddings.cols;  // حجم الـ Embedding

        for (int pos = 0; pos < seq_length; pos++) {
            for (int i = 0; i < embed_dim; i++) {
                // حساب المقام في المعادلة
                double denominator = pow(10000.0, (2.0 * (i / 2)) / embed_dim);
                
                // لو الرقم زوجي نستخدم Sin، لو فردي نستخدم Cos
                if (i % 2 == 0) {
                    seq_embeddings.data[pos][i] += sin(pos / denominator);
                } else {
                    seq_embeddings.data[pos][i] += cos(pos / denominator);
                }
            }
        }
    }
};

#endif