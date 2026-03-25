#include <iostream>

#include "los-transformers-de-la-city/data_preparation/data_preparation.h"
#include "los-transformers-de-la-city/layers/dense_layer.h"
#include "los-transformers-de-la-city/layers/layer.h"
#include "los-transformers-de-la-city/models/model.h"

int main() {
    Model model({5});

    std::cout << "test" << std::endl;
    model.addLayer(std::make_unique<DenseLayer>("dlayer0", 3));
    model.addLayer(std::make_unique<DenseLayer>("dlayer1", 1));
    model.print();
    Tensor<float> t({1, 5});
    t.at({0, 0}) = 1;
    t.at({0, 1}) = 7;
    t.at({0, 2}) = 1;
    t.at({0, 3}) = 2;
    t.at({0, 4}) = 3;

    std::cout << model.predict(t);
    return 0;
}