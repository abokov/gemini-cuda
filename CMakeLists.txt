cmake_minimum_required(VERSION 3.14)
project(gemini-cuda VERSION 1.0 LANGUAGES CXX)

# Require C++17 for modern memory management and filesystem features
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find system cURL
find_package(CURL REQUIRED)

# Automatically fetch nlohmann_json so users don't have to install it manually
include(FetchContent)
FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz
)
FetchContent_MakeAvailable(json)

# Define the executable
add_executable(gemini-cuda src/main.cpp)

# Link libraries
target_link_libraries(gemini-cuda PRIVATE 
    CURL::libcurl 
    nlohmann_json::nlohmann_json
)
