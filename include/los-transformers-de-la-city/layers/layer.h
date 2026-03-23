#pragma once

#include "los-transformers-de-la-city/math/linalg.h"

// includes base layer implementation
class Layer {
private:
    int m_input_dim;
    int m_output_dim;
    Layer* m_in_layer = nullptr;
    Layer* m_out_layer = nullptr;
    std::vector<float>* m_buffer_A;
    std::vector<float>* m_buffer_B;

public:
    Layer(const int input_dim, const int output_dim, std::vector<float>* buffer_A,
          std::vector<float>* buffer_B);

    const Tensor<float> gradient(const Tensor<float>& out_gradient);
    Tensor<float> pass(const Tensor<float>& input);
};