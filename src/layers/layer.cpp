#include "los-transformers-de-la-city/layers/layer.h"

Layer::Layer(const int input_dim, const int output_dim, std::vector<float>* buffer_A,
             std::vector<float>* buffer_B) {
    m_input_dim = input_dim;
    m_output_dim = output_dim;
    m_buffer_A = buffer_A;
    m_buffer_B = buffer_B;
}

const Tensor<float> Layer::gradient(const Tensor<float>& out_gradient) {
    Tensor<float> ret({});
    return ret;
}
Tensor<float> Layer::pass(const Tensor<float>& input) {
    Tensor<float> ret({});
    return ret;
}