#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <sstream>

using namespace std;

class Tokenizer {
private:
    unordered_map<string, int> word_to_id;
    unordered_map<int, string> id_to_word;
    int current_id = 1; // هنبدأ من 1 (0 ممكن نسيبه للـ Padding بعدين)

public:
    // دالة بتاخد جملة، وتقطعها، وتحفظ الكلمات الجديدة في القاموس
    void train(string text) {
        stringstream ss(text);
        string word;
        while (ss >> word) {
            if (word_to_id.find(word) == word_to_id.end()) {
                word_to_id[word] = current_id;
                id_to_word[current_id] = word;
                current_id++;
            }
        }
    }

    // تحويل النص لمصفوفة أرقام (عشان تدخلها للـ Neural Network)
    vector<int> encode(string text) {
        vector<int> ids;
        stringstream ss(text);
        string word;
        while (ss >> word) {
            if (word_to_id.find(word) != word_to_id.end()) {
                ids.push_back(word_to_id[word]);
            } else {
                // لو كلمة مش موجودة في القاموس، ممكن نديها ID خاص (مثلاً 0)
                ids.push_back(0); 
            }
        }
        return ids;
    }

    // تحويل الأرقام اللي طالعة من الـ Network لكلام نفهمه
    string decode(vector<int> ids) {
        string text = "";
        for (int id : ids) {
            if (id_to_word.find(id) != id_to_word.end()) {
                text += id_to_word[id] + " ";
            }
        }
        return text;
    }
};

#endif