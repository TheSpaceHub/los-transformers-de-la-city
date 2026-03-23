#pragma once

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

template <typename T>
class Tensor {
private:
    std::vector<int> m_dims;
    std::vector<int> m_strides;
    int m_rank;
    T* m_data;

public:
    // normal constructor
    Tensor(const std::vector<int>& dims) {
        m_rank = dims.size();
        m_dims = dims;
        int size = 1;
        m_strides.resize(m_rank);
        m_strides[m_rank - 1] = 1;
        for (size_t i = 0; i < m_rank; i++)
            size *= dims[i];
        for (size_t i = m_rank - 1; i > 0; i--)
            m_strides[i - 1] = m_strides[i] * dims[i];

        m_data = new T[size]();
    }
    // copy constructor
    Tensor(const Tensor& other) {
        m_dims = other.m_dims;
        m_strides = other.m_strides;
        m_rank = other.m_rank;
        int size = 1;
        for (size_t i = 0; i < m_rank; i++)
            size *= m_dims[i];
        m_data = new T[size];
        for (int i = 0; i < size; i++)
            m_data[i] = other.m_data[i];
    }

    // assignment
    Tensor& operator=(const Tensor& other) {
        if (this == &other)
            return *this;
        delete[] m_data;
        m_dims = other.m_dims;
        m_strides = other.m_strides;
        m_rank = other.m_rank;
        int size = 1;
        for (size_t i = 0; i < m_rank; i++)
            size *= m_dims[i];
        m_data = new T[size];
        for (int i = 0; i < size; i++)
            m_data[i] = other.m_data[i];
        return *this;
    }

    // get rid of the memory held in data
    ~Tensor() {
        delete[] m_data;
    }

    // accessing elements (write)
    T& at(const std::vector<int>& indeces) {
        int pos = 0;
        for (int i = 0; i < m_rank; i++)
            pos += m_strides[i] * indeces[i];
        return m_data[pos];
    }

    // accessing elements (read)
    T at(const std::vector<int>& indeces) const {
        int pos = 0;
        for (int i = 0; i < m_rank; i++)
            pos += m_strides[i] * indeces[i];
        return m_data[pos];
    }

    // getters for rows and cols
    int getDim(int d) const {
        return m_dims[d];
    }

    // equality
    friend bool operator==(const Tensor& a, const Tensor& b) {
        if (a.m_dims != b.m_dims)
            return false;
        int size = a.m_strides[0] * a.m_dims[0];
        for (int i = 0; i < size; i++)
            if (a.m_data[i] != b.m_data[i])
                return false;
        return true;
    }

    // printing tensors?
    friend std::ostream& operator<<(std::ostream& os, const Tensor& t) {
        std::cout << "(";
        for (size_t i = 0; i < t.m_rank - 1; i++)
            std::cout << t.m_dims[i] << ", ";
        std::cout << t.m_dims[t.m_rank - 1] << ")\n";
        // just print the whole block of data?
        for (size_t i = 0; i < t.m_dims[0] * t.m_strides[0]; i++)
            std::cout << t.m_data[i] << " ";
    };

    // add tensors
    Tensor<T> operator+(const Tensor<T>& t) const {
        std::cerr << "Warning: non-optimized tensor addition\n";
        Tensor<T> result(t.m_dims);

        for (size_t i = 0; i < t.m_dims[0] * t.m_strides[0]; i++)
            result.m_data[i] = m_data[i] + t.m_data[i];

        return result;
    }

    // substract tensors
    Tensor<T> operator-(const Tensor<T>& t) const {
        std::cerr << "Warning: non-optimized tensor substraction\n";
        Tensor<T> result(t.m_dims);

        for (size_t i = 0; i < t.m_dims[0] * t.m_strides[0]; i++)
            result.m_data[i] = m_data[i] - t.m_data[i];

        return result;
    }

    // tensor-scalar operations
    Tensor<T> operator*(const T n) const {
        Tensor<T> result(m_dims);
        for (size_t i = 0; i < m_dims[0] * m_strides[0]; i++)
            result.m_data[i] = m_data[i] * n;
        return result;
    }
    friend Tensor<T> operator*(const T n, const Tensor<T>& t) {
        return t * n;
    }
};