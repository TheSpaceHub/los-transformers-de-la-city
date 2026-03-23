#pragma once

#include "los-transformers-de-la-city/math/linalg.h"

// includes base layer implementation
class Layer {
private:
    std::vector<int> m_input_dims;
    std::vector<int> m_output_dims;
    Layer* m_in_layer = nullptr;
    Layer* m_out_layer = nullptr;
    std::vector<float>* m_buffer_A;
    std::vector<float>* m_buffer_B;
    std::string m_name = "";

public:
    Layer(std::string name, const std::vector<int>& input_dims,
          const std::vector<int>& output_dims = {}, std::vector<float>* buffer_A = nullptr,
          std::vector<float>* buffer_B = nullptr);
    virtual ~Layer() = default;

    virtual const Tensor<float> gradient(const Tensor<float>& out_gradient);
    virtual Tensor<float> pass(const Tensor<float>& input);
    virtual void initializeParameters();

    const std::vector<int> getInputDims();
    const std::vector<int> getOutputDims();
    const std::string getName();

    void setBuffers(std::vector<float>* buffer_A, std::vector<float>* buffer_B);
    void setOutputDims(const std::vector<int>& dims = {});
    virtual void setDefaultOutputDims();
};