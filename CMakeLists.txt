cmake_minimum_required(VERSION 3.15)
project(los-transformers-de-la-city VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MKL CONFIG REQUIRED)

add_executable(los_transformers_de_la_city
    src/main.cpp
    src/layers/layer.cpp
    src/layers/dense.cpp
    src/models/model.cpp
)

target_include_directories(los_transformers_de_la_city
    PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(los_transformers_de_la_city
    PRIVATE 
        MKL::MKL
)