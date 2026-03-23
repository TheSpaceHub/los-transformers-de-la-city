#include "los-transformers-de-la-city/models/model.h"

Model::Model() {}
void Model::fit(DataVec<float> dv) {}

// for now, only floats supported
std::vector<Tensor<float>> Model::predict() {
    std::vector<Tensor<float>> predictions = {};
    return predictions;
}