#include "los-transformers-de-la-city/layers/layer.h"

#include <stdexcept>

Layer::Layer(std::string name, const std::vector<int>& output_dims,
             const std::vector<int>& input_dims)
    : m_input_dims(input_dims), m_output_dims(output_dims), m_name(name) {
    if (m_output_dims.empty()) {
        throw std::runtime_error("Cannot set non-existent output");
    }
}

Tensor<float> Layer::gradient(const Tensor<float>& out_gradient) {
    return out_gradient;
}
Tensor<float> Layer::pass(const Tensor<float>& input) {
    return input;
}
void Layer::initializeParameters() {
    // nothing for default implementation
}

void Layer::print() {
    // outputs whatever is needed for debugging
    // nothing for default implementation
}