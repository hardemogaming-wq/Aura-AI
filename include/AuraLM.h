#ifndef AURALM_H
#define AURALM_H

#include "Tokenizer.h"
#include "Embedding.h"
#include "PositionalEncoding.h"
#include "Transformer.h" // اللي جواه TransformerBlock
#include "Layer.h"
#include <vector>
#include <string>
#include <fstream>

class AuraLM {
public:
    Tokenizer tokenizer;
    Embedding* embedding;
    TransformerBlock* block;
    Layer* lm_head;

    int vocab_size;
    int d_model;

    AuraLM(int vocab, int dim, int hidden_dim) {
        vocab_size = vocab;
        d_model = dim;
        
        embedding = new Embedding(vocab_size, d_model);
        block = new TransformerBlock(d_model, hidden_dim);
        
        // الـ LM Head: بياخد حجم الـ d_model ويطلعه لحجم القاموس كله
        // بنستخدم Softmax عشان نطلع احتمالات
        lm_head = new Layer(d_model, vocab_size, "softmax"); 
    }

    // الدالة اللي بتخلي الذكاء الاصطناعي يتكلم!
    std::string generate_next_word(std::string input_text) {
        // 1. تحويل النص لأرقام
        std::vector<int> tokens = tokenizer.encode(input_text);
        int seq_length = tokens.size();
        
        // 2. تحويل الأرقام لمصفوفة معاني (Embeddings)
        Matrix seq_matrix(seq_length, d_model);
        for(int i = 0; i < seq_length; i++) {
            Matrix word_vec = embedding->get_embedding(tokens[i]);
            for(int j = 0; j < d_model; j++) {
                seq_matrix.data[i][j] = word_vec.data[0][j];
            }
        }

        // 3. إضافة الـ GPS (Positional Encoding)
        PositionalEncoding::add_positional_encoding(seq_matrix);

        // 4. تمرير الجملة للـ Transformer Block ليفهم السياق
        Matrix context = block->forward(seq_matrix);

        // 5. استخراج "آخر كلمة" عشان نتوقع اللي بعدها
        Matrix last_word_context(d_model, 1);
        for(int j = 0; j < d_model; j++) {
            last_word_context.data[j][0] = context.data[seq_length - 1][j];
        }

        // 6. الـ LM Head بيطلع احتمالات القاموس كله
        Matrix probabilities = lm_head->forward(last_word_context);

        // 7. اختيار الكلمة صاحبة أعلى احتمال (Argmax)
        int best_word_id = 0;
        double max_prob = probabilities.data[0][0];
        for(int j = 1; j < vocab_size; j++) {
            if(probabilities.data[0][j] > max_prob) {
                max_prob = probabilities.data[0][j];
                best_word_id = j;
            }
        }

        // --- التعديل السحري للـ Debugging ---
        std::cout << "[Debug] المتوقع ID: " << best_word_id << " لـ Input: " << input_text << std::endl;
        
        // 8. فك التشفير وإرجاع الكلمة كنص
        std::vector<int> out_token = {best_word_id};
        std::string word = tokenizer.decode(out_token);
        
        if (word == "") {
            std::cout << "[Debug] تحذير: الكلمة فارغة!" << std::endl;
            return " <unknown> "; // نرجع دي عشان الـ Loop ميقفش
        }
        return word;
    }

    // دالة توليد نصوص طويلة (Autoregressive Generation)
    std::string generate_text(std::string seed_text, int max_words = 5) {
        std::string current_text = seed_text;

        for (int i = 0; i < max_words; i++) {
            // 1. نتوقع الكلمة الجاية بناءً على الجملة الحالية
            std::string next_word = generate_next_word(current_text);
            
            // 2. نتأكد إن في كلمة رجعت فعلاً
            if (next_word.empty()) {
                break; 
            }

            // 3. ندمج الكلمة الجديدة مع الجملة
            current_text += " " + next_word;
        }

        return current_text;
    }

    // دالة تدريب الموديل على جملة كاملة (Auto-regressive Training)
    void train_sentence(std::string sentence, int epochs = 100, double lr = 0.05) {
        std::vector<std::string> words;
        std::stringstream ss(sentence);
        std::string word;
        
        // تقطيع الجملة لكلمات
        while (ss >> word) {
            words.push_back(word);
        }

        // لو الجملة أقل من كلمتين، مفيش حاجة نتدرب عليها
        if (words.size() < 2) return;

        // لوب التدريب
        for (int e = 0; e < epochs; e++) {
            std::string current_context = words[0]; // نبدأ بأول كلمة
            
            for (size_t i = 1; i < words.size(); i++) {
                std::string target_word = words[i]; // الكلمة اللي المفروض يتوقعها
                
                // تحديث الأوزان
                train_step(current_context, target_word, lr);
                
                // نضيف الكلمة للسياق عشان يتوقع اللي بعدها في اللفة الجاية
                current_context += " " + target_word; 
            }
        }
    }

    // دالة التدريب لخطوة واحدة (تحديث الـ LM Head فقط حالياً)
    void train_step(std::string input_text, std::string target_word, double learning_rate = 0.01) {
        // 1. التمرير الأمامي (نفس اللي عملناه في الـ Generate)
        std::vector<int> tokens = tokenizer.encode(input_text);
        int seq_length = tokens.size();
        
        Matrix seq_matrix(seq_length, d_model);
        for(int i = 0; i < seq_length; i++) {
            Matrix word_vec = embedding->get_embedding(tokens[i]);
            for(int j = 0; j < d_model; j++) {
                seq_matrix.data[i][j] = word_vec.data[0][j];
            }
        }

        PositionalEncoding::add_positional_encoding(seq_matrix);
        Matrix context = block->forward(seq_matrix);

        Matrix last_word_context(d_model, 1);
        for(int j = 0; j < d_model; j++) {
            last_word_context.data[j][0] = context.data[seq_length - 1][j];
        }

        Matrix probabilities = lm_head->forward(last_word_context);

        // 2. حساب الخطأ (Gradient) للطبقة الأخيرة
        std::vector<int> target_tokens = tokenizer.encode(target_word);
        int target_id = target_tokens[0]; // هناخد الـ ID بتاع الكلمة الهدف

        Matrix d_output(vocab_size, 1); // مصفوفة الخطأ
        for(int j = 0; j < vocab_size; j++) {
            // المعادلة: (الاحتمال - 1 لو دي الكلمة الصح، أو الاحتمال - 0 لو كلمة غلط)
            double target_val = (j == target_id) ? 1.0 : 0.0;
            d_output.data[j][0] = probabilities.data[0][j] - target_val;
        }

        // 3. تحديث أوزان الـ LM Head بناءً على الخطأ
        // الـ weights بتنضرب في الـ Input مقلوب (last_word_context.transpose)
        Matrix weight_gradients = d_output.multiply(last_word_context.transpose());

        Matrix new_weights = lm_head->getWeights();
        for (int i = 0; i < new_weights.rows; i++) {
            for (int j = 0; j < new_weights.cols; j++) {
                new_weights.data[i][j] -= learning_rate * weight_gradients.data[i][j];
            }
        }
        lm_head->setWeights(new_weights);
        
        // تحديث الـ Biases لو عندك biases في الـ Layer
        Matrix new_bias = lm_head->getBias();
        for (int i = 0; i < new_bias.rows; i++) {
            new_bias.data[i][0] -= learning_rate * d_output.data[i][0];
        }
        lm_head->setBias(new_bias);
    }

    // دالة حفظ العقل (Weights) في ملف
    void save_model(std::string filename = "AuraLM_brain.txt") {
        std::ofstream out(filename);
        if (!out.is_open()) return;

        // نحفظ الأبعاد الأول عشان الأمان
        out << vocab_size << " " << d_model << "\n";
        
        // 1. حفظ أوزان الـ LM Head
        Matrix weights = lm_head->getWeights();
        for (int i = 0; i < weights.rows; i++) {
            for (int j = 0; j < weights.cols; j++) {
                out << weights.data[i][j] << " ";
            }
        }
        out << "\n";
        
        // 2. حفظ الـ Biases
        Matrix bias = lm_head->getBias();
        for (int i = 0; i < bias.rows; i++) {
            for (int j = 0; j < bias.cols; j++) {
                out << bias.data[i][j] << " ";
            }
        }
        out.close();
        std::cout << "💾 تم حفظ عقل AuraLM في [" << filename << "] بنجاح!\n";
    }

    // دالة استرجاع العقل من الملف
    bool load_model(std::string filename = "AuraLM_brain.txt") {
        std::ifstream in(filename);
        if (!in.is_open()) {
            std::cout << "⚠️ لم يتم العثور على ملف ذاكرة، سيتم البدء بعقل فارغ...\n";
            return false;
        }

        int v, d;
        in >> v >> d;
        // التأكد إن الموديل اللي في الملف متوافق مع الكود الحالي
        if (v != vocab_size || d != d_model) {
            std::cout << "❌ أبعاد النموذج في الملف لا تتطابق مع الإعدادات الحالية!\n";
            return false;
        }
        
        // 1. استرجاع الأوزان
        Matrix weights = lm_head->getWeights();
        for (int i = 0; i < weights.rows; i++) {
            for (int j = 0; j < weights.cols; j++) {
                in >> weights.data[i][j];
            }
        }
        lm_head->setWeights(weights);
        
        // 2. استرجاع الـ Biases
        Matrix bias = lm_head->getBias();
        for (int i = 0; i < bias.rows; i++) {
            for (int j = 0; j < bias.cols; j++) {
                in >> bias.data[i][j];
            }
        }
        lm_head->setBias(bias);
        in.close();
        std::cout << "🧠 تم استرجاع ذاكرة AuraLM من [" << filename << "] بنجاح، الموديل جاهز للعمل!\n";
        return true;
    }
};

#endif