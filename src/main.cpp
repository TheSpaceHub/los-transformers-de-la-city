#include <iostream>

#include "los-transformers-de-la-city/data_preparation/data_preparation.h"
#include "los-transformers-de-la-city/layers/layer.h"
#include "los-transformers-de-la-city/models/model.h"

int main() {
    Model model;
    model.addLayer(std::make_unique<Layer>("layer0", std::vector<int>{1}, std::vector<int>{1}));
    model.addLayer(std::make_unique<Layer>("layer1", std::vector<int>{1}, std::vector<int>{1}));
    model.print();
    return 0;
}