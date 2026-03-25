#pragma once

#include <vector>

#include "los-transformers-de-la-city/data_preparation/data_preparation.h"
#include "los-transformers-de-la-city/layers/layer.h"
#include "los-transformers-de-la-city/math/linalg.h"

class Model {
private:
    std::vector<std::unique_ptr<Layer>> m_layers = {};
    std::vector<int> m_input_dims = {};
    std::vector<int> m_output_dims = {};
    std::vector<float> m_buffer_A;
    std::vector<float> m_buffer_B;

    bool m_active_buffer = 0;  // 0: A,   1: B
    bool m_built = false;

    Tensor<float> pass(Tensor<float>& input);

public:
    Model(const std::vector<int>& input_dims);
    void fit(DataVec<float> dv);
    // for now, only floats supported
    Tensor<float> predict(Tensor<float> X);
    void build();
    void addLayer(std::unique_ptr<Layer> layer);
    void print();
};