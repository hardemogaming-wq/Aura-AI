#ifndef DATALOADER_H
#define DATALOADER_H

#include "Matrix.h"
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <stdexcept>

// دالة بتقرأ ملف الـ CSV وبترجع المدخلات والمخرجات
void load_csv(const std::string& filename, std::vector<Matrix>& inputs, std::vector<Matrix>& targets) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file: " + filename);
    }

    std::string line;
    std::getline(file, line); // تخطي أول سطر

    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string value;

        Matrix input(3, 1);
        Matrix target(1, 1);

        std::getline(ss, value, ','); input.data[0][0] = std::stod(value);
        std::getline(ss, value, ','); input.data[1][0] = std::stod(value);
        std::getline(ss, value, ','); input.data[2][0] = std::stod(value);
        std::getline(ss, value, ','); target.data[0][0] = std::stod(value);

        inputs.push_back(input);
        targets.push_back(target);
    }
}

#endif