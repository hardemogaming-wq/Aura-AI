#include "httplib.h"

#include "../include/NeuralNetwork.h"
#include <iostream>
#include <iomanip>
#include <limits>
#include <cmath>
#include <string>
#include <sstream>

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

std::string create_json_response(const std::string& diagnosis, double sugar, double bmi, double probability) {
    std::stringstream ss;
    ss << "{";
    ss << "\"diagnosis\":\"" << diagnosis << "\",";
    ss << "\"sugar\":" << sugar << ",";
    ss << "\"bmi\":" << bmi << ",";
    ss << "\"probability\":" << std::fixed << std::setprecision(4) << probability;
    ss << "}";
    return ss.str();
}

int main() {
    std::cout << "تشغيل خادم Aura-AI الطبي المحلي...\n";
    std::cout << "سيتم الاستماع على http://localhost:8080\n\n";

    // تحميل النموذج
    global_network = new NeuralNetwork();
    try {
        global_network->load_quantized_model("AuraModel_int8.bin");
        std::cout << "تم تحميل النموذج بنجاح من AuraModel_int8.bin\n";
    } catch (const std::exception& e) {
        std::cerr << "خطأ في تحميل النموذج: " << e.what() << std::endl;
        delete global_network;
        return 1;
    }

    httplib::Server svr;

    // نقطة نهاية للتشخيص
    svr.Get("/diagnose", [](const httplib::Request& req, httplib::Response& res) {
        if (global_network == nullptr) {
            res.status = 500;
            res.set_content("{\"error\":\"Model not loaded\"}", "application/json");
            return;
        }

        // قراءة المعاملات من الطلب
        double sugar = 0.0;
        double bmi = 0.0;

        if (req.has_param("sugar")) {
            try {
                sugar = std::stod(req.get_param_value("sugar"));
            } catch (...) {
                res.status = 400;
                res.set_content("{\"error\":\"Invalid sugar parameter\"}", "application/json");
                return;
            }
        }

        if (req.has_param("bmi")) {
            try {
                bmi = std::stod(req.get_param_value("bmi"));
            } catch (...) {
                res.status = 400;
                res.set_content("{\"error\":\"Invalid bmi parameter\"}", "application/json");
                return;
            }
        }

        // التحقق من صحة القيم
        if (sugar < 0 || sugar > 500 || bmi < 0 || bmi > 100) {
            res.status = 400;
            res.set_content("{\"error\":\"Invalid parameter values\"}", "application/json");
            return;
        }

        const double max_glucose = 200.0;
        const double max_bmi = 50.0;

        double normalized_sugar = sugar / max_glucose;
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
            res.status = 500;
            res.set_content("{\"error\":\"Invalid model input size\"}", "application/json");
            return;
        }

        Matrix output = global_network->forward(input);
        std::string diagnosis = diagnose_patient(output);
        double probability = output.data[0][0];

        std::string json_response = create_json_response(diagnosis, sugar, bmi, probability);
        res.set_content(json_response, "application/json");
    });

    // نقطة نهاية للتحقق من حالة الخادم
    svr.Get("/health", [](const httplib::Request& req, httplib::Response& res) {
        res.set_content("{\"status\":\"OK\",\"model_loaded\":" + std::string(global_network ? "true" : "false") + "}", "application/json");
    });

    // نقطة نهاية بسيطة
    svr.Get("/", [](const httplib::Request& req, httplib::Response& res) {
        res.set_content("Aura-AI Server Running. Use /diagnose?sugar=X&bmi=Y", "text/plain");
    });

    std::cout << "الخادم جاهز! افتح http://localhost:8080 في المتصفح\n";
    std::cout << "اضغط Ctrl+C لإيقاف الخادم\n";

    svr.listen("0.0.0.0", 8080);

    // تنظيف
    if (global_network) {
        delete global_network;
    }

    return 0;
}