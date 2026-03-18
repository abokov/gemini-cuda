#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdlib>
#include <curl/curl.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// cURL callback to write response data into a string
size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* userp) {
    userp->append((char*)contents, size * nmemb);
    return size * nmemb;
}

// Function to read the .cu file content
std::string read_file(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "[ERROR] Could not open file: " << filepath << std::endl;
        exit(1);
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

void audit_kernel(const std::string& kernel_code, const std::string& api_key) {
    // 1. Version-Agnostic Architecture: Pull Model ID from Environment
    const char* model_env = std::getenv("GEMINI_MODEL_ID");
    std::string model_id = model_env ? model_env : "gemini-1.5-pro";

    std::cout << "🚀 Dispatching architectural audit to: " << model_id << "...\n" << std::endl;

    // 2. Construct the production v1 endpoint
    std::string url = "https://generativelanguage.googleapis.com/v1/models/" + model_id + ":generateContent?key=" + api_key;

    // 3. Construct the JSON payload for the LLM
    json request_body = {
        {"contents", {{
            {"role", "user"},
            {"parts", {{
                {"text", "You are an expert NVIDIA CUDA Systems Engineer. "
                         "Perform a deep architectural audit of this kernel. "
                         "Focus strictly on memory coalescing, warp divergence, and race conditions (e.g., missing __syncthreads). "
                         "Do not output pleasantries. Output a strict audit report format.\n\n" + kernel_code}
            }}}
        }}}
    };

    std::string payload = request_body.dump();

    // 4. Initialize cURL and make the REST call
    CURL* curl = curl_easy_init();
    if (curl) {
        std::string response_string;
        struct curl_slist* headers = NULL;
        headers = curl_slist_append(headers, "Content-Type: application/json");

        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_string);

        CURLcode res = curl_easy_perform(curl);
        
        if (res != CURLE_OK) {
            std::cerr << "[ERROR] cURL failed: " << curl_easy_strerror(res) << std::endl;
        } else {
            // Parse and print the AI's response
            try {
                json response_json = json::parse(response_string);
                if (response_json.contains("candidates") && response_json["candidates"].size() > 0) {
                    std::string ai_text = response_json["candidates"][0]["content"]["parts"][0]["text"];
                    std::cout << "--- AUDIT REPORT ---\n" << ai_text << "\n--------------------\n";
                } else if (response_json.contains("error")) {
                    std::cerr << "[API ERROR] " << response_json["error"]["message"] << std::endl;
                } else {
                    std::cerr << "[ERROR] Unexpected API response format." << std::endl;
                }
            } catch (json::parse_error& e) {
                std::cerr << "[ERROR] Failed to parse JSON response: " << e.what() << std::endl;
            }
        }

        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: ./gemini-cuda <path_to_kernel.cu>" << std::endl;
        return 1;
    }

    const char* api_key_env = std::getenv("GEMINI_API_KEY");
    if (!api_key_env) {
        std::cerr << "[ERROR] GEMINI_API_KEY environment variable is not set." << std::endl;
        return 1;
    }

    std::string filepath = argv[1];
    std::string kernel_code = read_file(filepath);
    std::string api_key = api_key_env;

    audit_kernel(kernel_code, api_key);

    return 0;
}
