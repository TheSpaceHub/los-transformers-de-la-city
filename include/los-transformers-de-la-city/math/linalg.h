#pragma once

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

template <typename T>
class Tensor {
private:
    std::vector<T> m_memory;
    std::vector<int> m_dims;
    std::vector<int> m_strides;
    int m_rank;
    bool m_is_view;  // determines whether the tensor points to its own memory or something external
                     // (view)
    T* m_data_ptr;

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

        m_memory.resize(size);
        m_data_ptr = m_memory.data();
        m_is_view = false;
    }
    // view constructor
    Tensor(const std::vector<int>& dims, T* data_ptr) {
        m_rank = dims.size();
        m_dims = dims;
        int size = 1;
        m_strides.resize(m_rank);
        m_strides[m_rank - 1] = 1;
        for (size_t i = 0; i < m_rank; i++)
            size *= dims[i];
        for (size_t i = m_rank - 1; i > 0; i--)
            m_strides[i - 1] = m_strides[i] * dims[i];

        m_data_ptr = data_ptr;
        m_is_view = true;
    }
    // copy constructor
    Tensor(const Tensor& other) {
        m_dims = other.m_dims;
        m_strides = other.m_strides;
        m_rank = other.m_rank;
        m_memory = other.m_memory;
        m_data_ptr = (other.m_is_view) ? other.m_data_ptr : m_memory.data();
        m_is_view = other.m_is_view;
    }

    // assignment
    Tensor& operator=(const Tensor& other) {
        if (this == &other)
            return *this;
        m_dims = other.m_dims;
        m_strides = other.m_strides;
        m_rank = other.m_rank;
        m_memory = other.m_memory;
        m_data_ptr = (other.m_is_view) ? other.m_data_ptr : m_memory.data();
        m_is_view = other.m_is_view;
        return *this;
    }

    // get rid of the memory held in data
    ~Tensor() = default;

    // accessing elements (write)
    T& at(const std::vector<int>& indeces) {
        int pos = 0;
        for (int i = 0; i < m_rank; i++)
            pos += m_strides[i] * indeces[i];
        return m_data_ptr[pos];
    }

    // accessing elements (read)
    T at(const std::vector<int>& indeces) const {
        int pos = 0;
        for (int i = 0; i < m_rank; i++)
            pos += m_strides[i] * indeces[i];
        return m_data_ptr[pos];
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
            if (a.m_data_ptr[i] != b.m_data_ptr[i])
                return false;
        return true;
    }

    // printing tensors?
    friend std::ostream& operator<<(std::ostream& os, const Tensor& t) {
        os << "(";
        for (int i = 0; i < t.m_rank - 1; i++) {
            os << t.m_dims[i] << ", ";
        }
        os << t.m_dims[t.m_rank - 1] << ")\n";

        int total_size = t.m_dims[0] * t.m_strides[0];
        for (int i = 0; i < total_size; i++) {
            os << t.m_data_ptr[i];
            if (i < total_size - 1)
                os << " ";
        }
        os << "\n";
        return os;
    }

    // add tensors
    Tensor<T> operator+(const Tensor<T>& t) const {
        std::cerr << "Warning: non-optimized tensor addition\n";
        Tensor<T> result(t.m_dims);

        for (size_t i = 0; i < t.m_dims[0] * t.m_strides[0]; i++)
            result.m_data_ptr[i] = m_data_ptr[i] + t.m_data_ptr[i];

        return result;
    }

    // substract tensors
    Tensor<T> operator-(const Tensor<T>& t) const {
        std::cerr << "Warning: non-optimized tensor substraction\n";
        Tensor<T> result(t.m_dims);

        for (size_t i = 0; i < t.m_dims[0] * t.m_strides[0]; i++)
            result.m_data_ptr[i] = m_data_ptr[i] - t.m_data_ptr[i];

        return result;
    }

    // tensor-scalar operations
    Tensor<T> operator*(const T n) const {
        Tensor<T> result(m_dims);
        for (size_t i = 0; i < m_dims[0] * m_strides[0]; i++)
            result.m_data_ptr[i] = m_data_ptr[i] * n;
        return result;
    }
    friend Tensor<T> operator*(const T n, const Tensor<T>& t) {
        return t * n;
    }

    int rank() {
        return m_rank;
    }

    // special tensor operations that prevent moving more memory than necessary
    void reshape(const std::vector<int>& dims) {
        // reshapes the tensor without touching the data
        m_rank = dims.size();
        m_dims = dims;
        int size = 1;
        m_strides.resize(m_rank);
        m_strides[m_rank - 1] = 1;
        for (size_t i = 0; i < m_rank; i++)
            size *= dims[i];
        for (size_t i = m_rank - 1; i > 0; i--)
            m_strides[i - 1] = m_strides[i] * dims[i];
    }

    void permute(const std::vector<int>& perm) {
        // permutes dims
        // the user is trusted to provide a proper perm
        std::vector<int> new_dims(m_rank);
        std::vector<int> new_strides(m_rank);

        for (int i = 0; i < m_rank; i++) {
            new_dims[i] = m_dims[perm[i]];
            new_strides[i] = m_strides[perm[i]];
        }

        m_dims = std::move(new_dims);
        m_strides = std::move(new_strides);
    }

    T* data() {
        // returns the pointer to the data, careful when using
        return m_data_ptr;
    }
    const T* data() const {
        return m_data_ptr;
    }
};