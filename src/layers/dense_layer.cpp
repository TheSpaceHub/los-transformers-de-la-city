#include "los-transformers-de-la-city/layers/dense_layer.h"

#include <mkl.h>
#include <random>

void DenseLayer::initializeParameters() {
    std::random_device rd;
    std::mt19937 gen(rd());

    // xavier init
    float limit = std::sqrt(6.0f / (m_input_dims[0] + m_output_dims[0]));
    std::uniform_real_distribution<float> dist(-limit, limit);

    // must reset parameter structure
    m_weights = Tensor<float>({m_input_dims[0], m_output_dims[0]});
    m_biases = Tensor<float>({m_output_dims[0]});
    m_grad_weights = Tensor<float>({m_input_dims[0], m_output_dims[0]});
    m_grad_biases = Tensor<float>({m_output_dims[0]});
    m_cached_input = Tensor<float>(m_input_dims);

    float* weights_data = m_weights.data();
    float* biases_data = m_biases.data();

    for (int i = 0; i < m_input_dims[0] * m_output_dims[0]; ++i)
        weights_data[i] = dist(gen);

    for (int i = 0; i < m_output_dims[0]; ++i)
        biases_data[i] = 0;
}

Tensor<float> DenseLayer::pass(const Tensor<float>& input) {
    // expected input shape: (batch size, n)
    m_cached_input = input;
    std::vector<float>* out_buffer = m_buffer_A;

    if (input.data() == m_buffer_A->data()) {
        out_buffer = m_buffer_B;
    }

    int batch_size = input.getDim(0);
    int required_size = batch_size * m_output_dims[0];

    if (required_size > out_buffer->size()) {
        throw std::runtime_error(" Output exceeds pre-allocated buffer memory");
    }
    Tensor<float> output_view({batch_size, m_output_dims[0]}, out_buffer->data());

    int in_dim = input.getDim(1);
    int out_dim = m_weights.getDim(1);

    // weights (Y = alpha * X * W + beta * Y)
    cblas_sgemm(CblasRowMajor,                // strides for Row-Major memory
                CblasNoTrans,                 // do not transpose X
                CblasNoTrans,                 // do not transpose W
                batch_size, out_dim, in_dim,  // matrix dimensions
                1.0f,                         // alpha
                input.data(),                 // pointer to X memory
                in_dim,                       // stride of X
                m_weights.data(),             // pointer to W memory
                out_dim,                      // stride of W
                0.0f,                         // beta
                output_view.data(),           // pointer to Y memory
                out_dim                       // stride of Y
    );

    // bias

    float* out_data = output_view.data();
    float* bias_data = m_biases.data();

    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < out_dim; j++) {
            out_data[i * out_dim + j] += bias_data[j];
        }
    }

    return output_view;
}

Tensor<float> DenseLayer::gradient(const Tensor<float>& out_gradient) {
    return out_gradient;
}

void DenseLayer::print() {
    std::cout << "Weights: " << m_weights << " ";
    std::cout << "Biases: " << m_biases << " ";
}