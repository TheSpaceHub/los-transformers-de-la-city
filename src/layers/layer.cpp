#include "los-transformers-de-la-city/layers/layer.h"

#include <stdexcept>

Layer::Layer(std::string name, const std::vector<int>& input_dims,
             const std::vector<int>& output_dims, std::vector<float>* buffer_A,
             std::vector<float>* buffer_B) {
    if (input_dims.size() == 0)
        std::runtime_error("Cannot set non-existant input");
    m_input_dims = input_dims;
    m_output_dims = output_dims;
    m_buffer_A = buffer_A;
    m_buffer_B = buffer_B;
    m_name = name;
}

const Tensor<float> Layer::gradient(const Tensor<float>& out_gradient) {
    Tensor<float> ret({});
    return ret;
}
Tensor<float> Layer::pass(const Tensor<float>& input) {
    Tensor<float> ret({});
    return ret;
}
void Layer::initializeParameters() {
    // nothing for default
}

const std::vector<int> Layer::getInputDims() {
    return m_input_dims;
}
const std::vector<int> Layer::getOutputDims() {
    return m_output_dims;
}
const std::string Layer::getName() {
    return m_name;
}

void Layer::setBuffers(std::vector<float>* buffer_A, std::vector<float>* buffer_B) {
    m_buffer_A = buffer_A;
    m_buffer_B = buffer_B;
}
void Layer::setOutputDims(const std::vector<int>& dims) {
    m_output_dims = dims;
}
void Layer::setDefaultOutputDims() {
    m_output_dims = m_input_dims;
}