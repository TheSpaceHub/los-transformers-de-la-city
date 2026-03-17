#include <iostream>

#include "los-transformers-de-la-city/data_preparation/data_preparation.h"

int main() {
    std::cout << "Hello, World!" << std::endl;
    DataVec<std::string> dv = DataVec<std::string>::readCSV("data/wow.csv");
    return 0;
}