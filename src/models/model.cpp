#include "los-transformers-de-la-city/models/model.h"

#include <iostream>

#define DEFAULT_BUFFER_SIZE_LOG_2 10

Model::Model() {
    m_buffer_A.resize(1 << DEFAULT_BUFFER_SIZE_LOG_2);
    m_buffer_B.resize(1 << DEFAULT_BUFFER_SIZE_LOG_2);
}

void Model::fit(DataVec<float> dv) {}

std::vector<Tensor<float>> Model::predict(std::vector<Tensor<float>> X) {
    std::vector<Tensor<float>> predictions = {};
    for (auto& element : X)
        predictions.push_back(pass(element));

    return predictions;
}

void Model::build() {
    for (size_t i = 0; i < m_layers.size() - 1; i++) {
        if (m_layers[i]->getOutputDims().size() == 0) {
            // output has not been set
            m_layers[i]->setOutputDims(m_layers[i + 1]->getInputDims());
        }
        m_layers[i]->initializeParameters();
    }
    // if the last layer does not have output dims set, we set them to the layer's default
    if (m_layers[m_layers.size() - 1]->getOutputDims().size() == 0)
        m_layers[m_layers.size() - 1]->setDefaultOutputDims();
    m_layers[m_layers.size() - 1]->initializeParameters();
    m_built = true;
}

void Model::addLayer(std::unique_ptr<Layer> layer) {
    // being the last layer, this is the current output
    m_output_dims = layer->getOutputDims();
    if (m_layers.size() == 0) {
        // the layer is the input layer
        m_input_dims = layer->getInputDims();
    }
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
                std::cout << ")\n";
        }
    }
}

Tensor<float> Model::pass(Tensor<float>& input) {
    if (!m_built)
        build();
    return Tensor<float>(std::vector<int>{});
}