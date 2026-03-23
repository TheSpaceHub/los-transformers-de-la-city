#pragma once

#include <vector>

#include "los-transformers-de-la-city/data_preparation/data_preparation.h"
#include "los-transformers-de-la-city/layers/layer.h"
#include "los-transformers-de-la-city/math/linalg.h"

class Model {
private:
    std::vector<Layer> layers = {};
public:
    Model();
    void fit(DataVec<float> dv);
    // for now, only floats supported
    std::vector<Tensor<float>> predict();
};