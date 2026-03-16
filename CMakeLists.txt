cmake_minimum_required(VERSION 3.15)
project(los-transformers-de-la-city VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(data_preparation_lib
    src/data_preparation/data_preparation.cpp
)

target_include_directories(data_preparation_lib
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}/src/data_preparation/
)

add_executable(los_transformers_de_la_city
    src/main.cpp
)

target_link_libraries(los_transformers_de_la_city
    PRIVATE 
        data_preparation_lib
)