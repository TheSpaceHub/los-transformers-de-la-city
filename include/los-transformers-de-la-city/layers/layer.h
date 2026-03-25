#pragma once

#include "los-transformers-de-la-city/math/linalg.h"

// includes base layer implementation
class Layer {
protected:
    std::vector<int> m_input_dims;
    std::vector<int> m_output_dims;
    std::vector<float>* m_buffer_A = nullptr;
    std::vector<float>* m_buffer_B = nullptr;
    std::string m_name = "";

public:
    Layer(std::string name, const std::vector<int>& output_dims,
          const std::vector<int>& input_dims = {});
    virtual ~Layer() = default;

    virtual Tensor<float> gradient(const Tensor<float>& out_gradient);
    virtual Tensor<float> pass(const Tensor<float>& input);
    virtual void initializeParameters();
    virtual void print();

    const std::vector<int> getInputDims() const {
        return m_input_dims;
    }
    const std::vector<int> getOutputDims() const {
        return m_output_dims;
    }
    const std::string getName() const {
        return m_name;
    }

    void setBuffers(std::vector<float>* buffer_A, std::vector<float>* buffer_B) {
        m_buffer_A = buffer_A;
        m_buffer_B = buffer_B;
    }
    void setInputDims(const std::vector<int>& dims = {}) {
        m_input_dims = dims;
    }
    void setOutputDims(const std::vector<int>& dims = {}) {
        m_output_dims = dims;
    }
    virtual void setDefaultOutputDims() {
        m_output_dims = m_input_dims;
    }
};