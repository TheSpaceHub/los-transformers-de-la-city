#include "los-transformers-de-la-city/models/model.h"

#include <iostream>

#define DEFAULT_BUFFER_SIZE_LOG_2 10

Model::Model(const std::vector<int>& input_dims) {
    m_buffer_A.resize(1 << DEFAULT_BUFFER_SIZE_LOG_2);
    m_buffer_B.resize(1 << DEFAULT_BUFFER_SIZE_LOG_2);
    m_input_dims = input_dims;
}

void Model::fit(DataVec<float> dv) {}

Tensor<float> Model::predict(Tensor<float> X) {
    return pass(X);
}

void Model::build() {
    for (size_t i = 1; i < m_layers.size(); i++) {
        if (m_layers[i]->getInputDims().empty()) {
            // input has not been set
            m_layers[i]->setInputDims(m_layers[i - 1]->getOutputDims());
        }
        m_layers[i]->initializeParameters();
    }
    m_built = true;
}

void Model::addLayer(std::unique_ptr<Layer> layer) {
    if (m_layers.empty()) {
        // the layer is the input layer
        if (m_input_dims.empty())
            m_input_dims = layer->getInputDims();
        else
            layer->setInputDims(m_input_dims);
    } else {
        // new layer's input is last one's output
        layer->setInputDims(m_output_dims);
    }

    // being the last layer, this is the current output
    m_output_dims = layer->getOutputDims();

    layer->initializeParameters();

    m_layers.push_back(std::move(layer));

    // set buffers in layer
    m_layers[m_layers.size() - 1]->setBuffers(&m_buffer_A, &m_buffer_B);
}

void Model::print() {
    // prints out the structure of the model
    for (size_t i = 0; i < m_layers.size(); i++) {
        std::cout << "Layer " << i << ": " << m_layers[i]->getName();
        int padding = 30 - m_layers[i]->getName().size();
        for (size_t j = 0; j < padding; j++)
            std::cout << " ";
        std::cout << "Shape: (";
        for (size_t j = 0; j < m_layers[i]->getInputDims().size(); j++) {
            std::cout << m_layers[i]->getInputDims()[j];
            if (j < m_layers[i]->getInputDims().size() - 1)
                std::cout << ", ";
            else
                std::cout << ") -> (";
        }
        for (size_t j = 0; j < m_layers[i]->getOutputDims().size(); j++) {
            std::cout << m_layers[i]->getOutputDims()[j];
            if (j < m_layers[i]->getOutputDims().size() - 1)
                std::cout << ", ";
            else
                std::cout << ")\n";
        }
        // print layer if it has anything to say
        m_layers[i]->print();
    }
}

Tensor<float> Model::pass(Tensor<float>& input) {
    if (!m_built)
        build();
    Tensor<float> result = input;
    for (size_t i = 0; i < m_layers.size(); i++) {
        result = m_layers[i]->pass(result);
    }
    return result;
}