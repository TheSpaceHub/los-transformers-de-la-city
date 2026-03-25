#pragma once

#include "los-transformers-de-la-city/layers/layer.h"

class DenseLayer : public Layer {
protected:
    Tensor<float> m_weights;
    Tensor<float> m_biases;
    Tensor<float> m_grad_weights;
    Tensor<float> m_grad_biases;
    Tensor<float> m_cached_input;

public:
    DenseLayer(std::string name, int output_dim)
        : Layer(name, {output_dim}),

          m_weights({1, output_dim}),
          m_biases({output_dim}),
          m_grad_weights({1, output_dim}),
          m_grad_biases({output_dim}),

          m_cached_input({0}) {}

    void initializeParameters() override;

    Tensor<float> pass(const Tensor<float>& input) override;
    Tensor<float> gradient(const Tensor<float>& out_gradient) override;
    void print() override;

    Tensor<float>& getWeights() {
        return m_weights;
    }
    Tensor<float>& getBiases() {
        return m_biases;
    }
    Tensor<float>& getWeightGradients() {
        return m_grad_weights;
    }
    Tensor<float>& getBiasGradients() {
        return m_grad_biases;
    }
};