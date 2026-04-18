#include "../include/NeuralNetwork.h"
#include "../include/Loss.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <cmath>
#include <ctime>
#include <iomanip>

int main() {
    const int sample_count = 5000;
    const double max_glucose = 200.0;
    const double max_bmi = 50.0;

    std::mt19937 rng(static_cast<unsigned>(std::time(nullptr)));
    std::uniform_real_distribution<double> glucose_dist(70.0, 180.0);
    std::uniform_real_distribution<double> bmi_dist(15.0, 50.0);

    std::vector<Matrix> inputs;
    std::vector<Matrix> targets;
    inputs.reserve(sample_count);
    targets.reserve(sample_count);

    std::ofstream csv_file("synthetic_medical_data.csv");
    csv_file << "glucose,bmi,label\n";

    for (int i = 0; i < sample_count; ++i) {
        double glucose = glucose_dist(rng);
        double bmi = bmi_dist(rng);
        double normalized_glucose = glucose / max_glucose;
        double normalized_bmi = bmi / max_bmi;

        double score = normalized_glucose * 0.6 + normalized_bmi * 0.4;
        double label = score > 0.55 ? 1.0 : 0.0;

        Matrix input(2, 1);
        input.data[0][0] = normalized_glucose;
        input.data[1][0] = normalized_bmi;
        inputs.push_back(input);

        Matrix target(1, 1);
        target.data[0][0] = label;
        targets.push_back(target);

        csv_file << glucose << "," << bmi << "," << label << "\n";
    }

    csv_file.close();
    std::cout << "تم إنشاء قاعدة بيانات اصطناعية مكونة من " << sample_count << " مريض.\n";
    std::cout << "ملف البيانات: synthetic_medical_data.csv\n\n";

    NeuralNetwork network({2, 10, 1}, {"relu", "sigmoid"}, 0.02);
    const int epochs = 1200;

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        double epoch_loss = 0.0;
        for (int i = 0; i < sample_count; ++i) {
            Matrix output = network.forward(inputs[i]);
            epoch_loss += Loss::mse_loss(output, targets[i]);
            network.backward(inputs[i], targets[i], 0.01);
        }
        epoch_loss /= sample_count;
        if (epoch % 100 == 0) {
            std::cout << "Epoch " << epoch << " - متوسط الخطأ: " << std::fixed << std::setprecision(4) << epoch_loss << "\n";
        }
    }

    int correct = 0;
    for (int i = 0; i < sample_count; ++i) {
        Matrix output = network.forward(inputs[i]);
        int predicted = output.data[0][0] >= 0.5 ? 1 : 0;
        int actual = targets[i].data[0][0] >= 0.5 ? 1 : 0;
        if (predicted == actual) {
            ++correct;
        }
    }

    double accuracy = 100.0 * correct / sample_count;
    std::cout << "\nدقة النموذج بعد التدريب: " << std::fixed << std::setprecision(2) << accuracy << "%\n";

    network.save_model("model_weights_synthetic.txt");
    network.save_quantized_model("AuraModel_int8.bin");
    std::cout << "تم حفظ النموذج النهائي إلى model_weights_synthetic.txt و AuraModel_int8.bin\n";

    return 0;
}
