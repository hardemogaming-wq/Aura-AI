#include "../include/Matrix.h"
#include "../include/Layer.h"
#include "../include/NeuralNetwork.h"
#include "../include/Activation.h"
#include "../include/Loss.h"
#include "../include/Transformer.h"
#include "../include/Tokenizer.h"
#include "../include/Embedding.h"
#include "../include/PositionalEncoding.h"
#include "../include/LayerNorm.h"
#include "../include/FeedForward.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <limits>

using namespace std;

static void clear_terminal() {
    cout << "\033[2J\033[H";
}

static double prompt_double(const string& prompt) {
    double value;
    while (true) {
        cout << prompt;
        if (cin >> value) {
            return value;
        }
        cin.clear();
        cin.ignore(numeric_limits<streamsize>::max(), '\n');
        cout << "الرجاء إدخال قيمة رقمية صحيحة.\n";
    }
}

static string diagnose_patient(const Matrix& output) {
    if (output.rows < 1 || output.cols < 1) {
        return "خطأ في ناتج النموذج.";
    }

    double probability = output.data[0][0];
    int percent = static_cast<int>(round(probability * 100.0));
    if (percent < 0) percent = 0;
    if (percent > 100) percent = 100;

    if (probability >= 0.5) {
        return "التشخيص: المريض مصاب بالسكري بنسبة " + to_string(percent) + "%";
    }
    return "التشخيص: المريض سليم.";
}

int main() {
    clear_terminal();
    cout << "أهلاً بك في دكتور Aura-AI 🤖" << endl;
    cout << "نظام تشخيص بسيط يستخدم نموذجاً محفوظاً في model_weights.txt" << endl << endl;

    double sugar_level = prompt_double("أدخل نسبة السكر: ");
    double bmi = prompt_double("أدخل مؤشر كتلة الجسم (BMI): ");
    double age = prompt_double("أدخل العمر: ");

    const double max_glucose = 200.0;
    const double max_bmi = 50.0;
    const double max_age = 100.0;

    try {
        vector<int> layer_sizes = {3, 4, 2};
        vector<string> activations = {"relu", "sigmoid"};
        NeuralNetwork network(layer_sizes, activations);
        network.load_model("model_weights.txt");

        cout << "\n✓ تم تحميل النموذج بنجاح من model_weights.txt" << endl;

        Matrix input(3, 1);
        input.data[0][0] = sugar_level / max_glucose;
        input.data[1][0] = bmi / max_bmi;
        input.data[2][0] = age / max_age;

        Matrix output = network.forward(input);

        cout << "\nالمدخلات:" << endl;
        input.print();
        cout << "\nالمخرجات (احتمالية):" << endl;
        output.print();
        cout << "\n" << diagnose_patient(output) << endl;

        network.save_quantized_model("AuraModel_int8.bin");
        std::ifstream bin_file("AuraModel_int8.bin", std::ios::binary | std::ios::ate);
        if (bin_file.is_open()) {
            std::streamsize file_size = bin_file.tellg();
            cout << "\nتم حفظ النموذج المضغوط في AuraModel_int8.bin (" << file_size << " bytes)." << endl;
            bin_file.close();
        } else {
            cout << "\nتم حفظ النموذج المضغوط في AuraModel_int8.bin." << endl;
        }

    } catch (const std::runtime_error& e) {
        cerr << "خطأ: " << e.what() << endl;
        return 1;
    } catch (const exception& e) {
        cerr << "حدث خطأ غير متوقع: " << e.what() << endl;
        return 1;
    }

    // اختبار Self-Attention
    cout << "\n🧠 اختبار Self-Attention:" << endl;
    SelfAttention attention;

    // إنشاء مصفوفات وهمية 3x3
    Matrix Q(3, 3);
    Matrix K(3, 3);
    Matrix V(3, 3);

    // ملء المصفوفات بقيم وهمية
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            Q.data[i][j] = i + j + 1.0;
            K.data[i][j] = i - j + 1.0;
            V.data[i][j] = i * j + 1.0;
        }
    }

    cout << "Q (Query):" << endl;
    Q.print();
    cout << "K (Key):" << endl;
    K.print();
    cout << "V (Value):" << endl;
    V.print();

    Matrix attention_output = attention.compute_attention(Q, K, V);
    cout << "نتيجة Self-Attention:" << endl;
    attention_output.print();

    // اختبار الـ Tokenizer
    cout << "\n🔠 اختبار الـ Tokenizer:" << endl;
    Tokenizer tokenizer;

    // تدريب على جملة عربية وإنجليزية
    string training_text = "أنا مهندس ذكاء اصطناعي I am an AI engineer";
    tokenizer.train(training_text);
    cout << "تم التدريب على: " << training_text << endl;

    // Encode
    vector<int> encoded = tokenizer.encode(training_text);
    cout << "الـ Encode (الأرقام): ";
    for (int id : encoded) {
        cout << id << " ";
    }
    cout << endl;

    // Decode
    string decoded = tokenizer.decode(encoded);
    cout << "الـ Decode (النص): " << decoded << endl;

    // اختبار الـ Embedding
    cout << "\n🌌 اختبار الـ Embedding:" << endl;
    Embedding embedding(10, 3); // حجم القاموس 10، حجم الـ Embedding 3
    cout << "تم إنشاء Embedding مع vocab_size=10, embedding_dim=3" << endl;

    // جلب معنى كلمة "أنا" (ID=1)
    Matrix word_vector = embedding.get_embedding(1);
    cout << "معنى كلمة 'أنا' (ID=1):" << endl;
    word_vector.print();

    // اختبار الـ Positional Encoding
    cout << "\n📍 اختبار الـ Positional Encoding:" << endl;
    Matrix seq_embeddings(3, 3); // 3 كلمات، embedding dim 3
    // ملء بأرقام عشوائية
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            seq_embeddings.data[i][j] = ((double) rand() / (RAND_MAX)) * 0.1;
        }
    }
    cout << "المصفوفة قبل إضافة Positional Encoding:" << endl;
    seq_embeddings.print();

    PositionalEncoding::add_positional_encoding(seq_embeddings);
    cout << "المصفوفة بعد إضافة Positional Encoding:" << endl;
    seq_embeddings.print();

    // اختبار الـ Layer Normalization
    cout << "\n⚖️ اختبار الـ Layer Normalization:" << endl;
    LayerNorm layer_norm;
    layer_norm.forward(seq_embeddings);
    cout << "المصفوفة بعد تطبيق Layer Normalization:" << endl;
    seq_embeddings.print();

    // اختبار الـ Feed-Forward Network
    cout << "\n⚙️ اختبار الـ Feed-Forward Network:" << endl;
    FeedForward ffn(3, 12); // d_model=3, d_ff=12
    Matrix ffn_output = ffn.forward(seq_embeddings);
    cout << "المصفوفة بعد تطبيق Feed-Forward Network:" << endl;
    ffn_output.print();

    return 0;
}
