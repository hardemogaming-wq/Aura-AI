#include "../include/NeuralNetwork.h"
#include <iostream>
#include <iomanip>
#include <limits>
#include <cmath>
#include <string>

static NeuralNetwork* global_network = nullptr;

static std::string diagnose_patient(const Matrix& output) {
    if (output.rows < 1 || output.cols < 1) {
        return "خطأ في ناتج النموذج.";
    }

    double probability = output.data[0][0];
    int percent = static_cast<int>(std::round(probability * 100.0));
    if (percent < 0) percent = 0;
    if (percent > 100) percent = 100;

    if (probability >= 0.5) {
        return "التشخيص: المريض مصاب بالسكري بنسبة " + std::to_string(percent) + "%";
    }
    return "التشخيص: المريض سليم.";
}

extern "C" {

const char* init_aura_model() {
    if (global_network == nullptr) {
        global_network = new NeuralNetwork();
        try {
            global_network->load_quantized_model("AuraModel_int8.bin");
            return "Model loaded successfully";
        } catch (const std::exception& e) {
            delete global_network;
            global_network = nullptr;
            return "Failed to load model";
        }
    }
    return "Model already loaded";
}

const char* diagnose_diabetes(double sugar_level, double bmi) {
    if (global_network == nullptr) {
        return "Model not loaded";
    }

    const double max_glucose = 200.0;
    const double max_bmi = 50.0;

    double normalized_sugar = sugar_level / max_glucose;
    double normalized_bmi = bmi / max_bmi;

    if (normalized_sugar < 0.0) normalized_sugar = 0.0;
    if (normalized_sugar > 1.0) normalized_sugar = 1.0;
    if (normalized_bmi < 0.0) normalized_bmi = 0.0;
    if (normalized_bmi > 1.0) normalized_bmi = 1.0;

    int input_size = global_network->getLayer(0).getInputSize();
    Matrix input(input_size, 1);
    if (input_size == 2) {
        input.data[0][0] = normalized_sugar;
        input.data[1][0] = normalized_bmi;
    } else if (input_size == 3) {
        input.data[0][0] = normalized_sugar;
        input.data[1][0] = normalized_bmi;
        input.data[2][0] = 1.0;
    } else {
        return "Invalid model input size";
    }

    Matrix output = global_network->forward(input);
    static std::string result = diagnose_patient(output);
    return result.c_str();
}

void cleanup_aura_model() {
    if (global_network != nullptr) {
        delete global_network;
        global_network = nullptr;
    }
}

}

