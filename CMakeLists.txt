cmake_minimum_required(VERSION 3.14)
project(llm-cuda-auditor VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# --- NEW: Build Options ---
option(USE_CLAUDE "Build with Anthropic Claude support instead of Gemini" OFF)

find_package(CURL REQUIRED)

include(FetchContent)
FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz)
FetchContent_MakeAvailable(json)

# You might want to rename the executable to be engine-agnostic
add_executable(cuda-audit src/main.cpp)

# --- NEW: Inject the Preprocessor Directive ---
if(USE_CLAUDE)
    target_compile_definitions(cuda-audit PRIVATE USE_CLAUDE)
    message(STATUS "Building with Anthropic Claude backend.")
else()
    message(STATUS "Building with Google Gemini backend.")
endif()

target_link_libraries(cuda-audit PRIVATE CURL::libcurl nlohmann_json::nlohmann_json)

