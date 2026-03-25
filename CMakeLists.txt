cmake_minimum_required(VERSION 3.15)
project(los-transformers-de-la-city VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MKL CONFIG REQUIRED)

add_executable(los-transformers-de-la-city
    src/main.cpp
    src/layers/layer.cpp
    src/layers/dense_layer.cpp
    src/models/model.cpp
)

target_include_directories(los-transformers-de-la-city
    PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(los-transformers-de-la-city
    PRIVATE 
        MKL::MKL
)