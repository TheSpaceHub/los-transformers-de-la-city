#pragma once
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

template <typename T>
class DataVec
/* Object that holds data in a tabular manner. Similar to a DataFrame in pandas
   except that this is more rigid in regard to types.*/
{
private:
    std::vector<std::string> m_columns = {};
    std::map<std::string, int> m_col_to_index;

    std::vector<std::vector<T>> m_data;

    static T stringToT(const std::string& str) {
        // must implement string to type conversion if you want to use the type in a
        // datavec
        if constexpr (std::is_same_v<T, std::string>) {
            return str;
        } else if constexpr (std::is_same_v<T, int>) {
            return std::stoi(str);
        } else if constexpr (std::is_same_v<T, double>) {
            return std::stod(str);
        } else if constexpr (std::is_same_v<T, float>) {
            return std::stof(str);
        } else if constexpr (std::is_same_v<T, bool>) {
            return (str == "true" || str == "1");
        } else
            throw std::runtime_error("The conversion to the desired type has not been specified.");
    }

public:
    DataVec(std::vector<std::string> columns = {}) {
        m_columns = columns;
        for (size_t i = 0; i < columns.size(); i++) {
            if (m_col_to_index.find(columns[i]) != m_col_to_index.end()) {
                throw std::runtime_error("Cannot define columns with the same name.");
            } else
                m_col_to_index[columns[i]] = i;
        }
        m_data.resize(m_columns.size());
    }
    ~DataVec() {}

    static DataVec<T> readCSV(std::string relative_file_path, char separator = ',') {
        fs::path file_path = fs::current_path() / relative_file_path;

        // check if the file exists
        if (!fs::exists(file_path) || !fs::is_regular_file(file_path)) {
            std::cerr << "Invalid file path: could not read " + file_path.string() + "\n";
            return DataVec<T>();
        }

        // read the file
        std::ifstream file_stream(file_path);

        if (!file_stream.is_open()) {
            std::cerr << "Failed to open file: " << file_path.string() << "\n";
            return DataVec<T>();
        }

        std::vector<std::string> columns = {};
        std::vector<std::vector<T>> data = {};

        std::string line;
        bool first_line_read = false;
        while (std::getline(file_stream, line)) {
            if (!first_line_read) {
                std::string char_stack = "";
                for (size_t i = 0; i < line.size(); i++) {
                    if (line[i] == separator) {
                        // store accumulated value
                        columns.push_back(char_stack);
                        char_stack = "";
                    } else
                        char_stack += line[i];
                }
                // store whatever is left
                columns.push_back(char_stack);

                // set columns and data
                data.resize(columns.size());
            } else {
                size_t element_count = 0;
                std::string char_stack = "";
                for (size_t i = 0; i < line.size(); i++) {
                    if (line[i] == separator) {
                        // store accumulated value
                        data[element_count].push_back(stringToT(char_stack));
                        char_stack = "";
                        element_count++;
                        if (element_count == columns.size())
                            throw std::runtime_error("Too many columns in row " +
                                                     std::to_string((int)data[0].size()));
                    } else
                        char_stack += line[i];
                }
                // store whatever is left
                data[element_count].push_back(stringToT(char_stack));
                if (element_count < columns.size() - 1)
                    throw std::runtime_error("Not enough columns in row " +
                                             std::to_string((int)data[0].size()));
            }
            std::cout << "-" + line + "-" << "\n";
        }

        DataVec<T> dv(columns);
        dv.m_data = std::move(data);
        return dv;
    }

    std::vector<T>& at(const std::string& column) {
        return m_data[m_col_to_index[column]];
    }
};