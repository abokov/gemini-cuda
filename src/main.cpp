#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdlib>
#include <curl/curl.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* userp) {
    userp->append((char*)contents, size * nmemb);
    return size * nmemb;
}

std::string read_file(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) exit(1);
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

void audit_kernel(const std::string& kernel_code, const std::string& api_key) {
    std::string url;
    std::string payload;
    struct curl_slist* headers = NULL;
    headers = curl_slist_append(headers, "Content-Type: application/json");

    const std::string system_prompt = "You are an expert NVIDIA CUDA Systems Engineer. "
                                      "Perform a deep architectural audit of this kernel. "
                                      "Focus on memory coalescing, warp divergence, and race conditions.";

#ifdef USE_CLAUDE
    // ==========================================
    // ANTHROPIC CLAUDE BACKEND
    // ==========================================
    std::cout << "🚀 Dispatching audit to: Claude 3.5 Sonnet...\n" << std::endl;
    
    url = "https://api.anthropic.com/v1/messages";
    
    // Claude requires specific headers
    std::string key_header = "x-api-key: " + api_key;
    headers = curl_slist_append(headers, key_header.c_str());
    headers = curl_slist_append(headers, "anthropic-version: 2023-06-01");

    json request_body = {
        {"model", "claude-3-5-sonnet-20241022"},
        {"max_tokens", 1024},
        {"system", system_prompt},
        {"messages", {{
            {"role", "user"},
            {"content", kernel_code}
        }}}
    };
    payload = request_body.dump();

#else
    // ==========================================
    // GOOGLE GEMINI BACKEND (Default)
    // ==========================================
    const char* model_env = std::getenv("GEMINI_MODEL_ID");
    std::string model_id = model_env ? model_env : "gemini-3.1-pro";
    
    std::cout << "🚀 Dispatching audit to: " << model_id << "...\n" << std::endl;
    
    url = "https://generativelanguage.googleapis.com/v1/models/" + model_id + ":generateContent?key=" + api_key;

    json request_body = {
        {"contents", {{
            {"role", "user"},
            {"parts", {{
                {"text", system_prompt + "\n\n" + kernel_code}
            }}}
        }}}
    };
    payload = request_body.dump();
#endif

    // --- Execute cURL (Same for both backends) ---
    CURL* curl = curl_easy_init();
    if (curl) {
        std::string response_string;
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_string);

        CURLcode res = curl_easy_perform(curl);
        
        if (res == CURLE_OK) {
            try {
                json response_json = json::parse(response_string);
                
#ifdef USE_CLAUDE
                // Parse Claude Response
                if (response_json.contains("content")) {
                    std::cout << "--- AUDIT REPORT ---\n" 
                              << response_json["content"][0]["text"].get<std::string>() 
                              << "\n--------------------\n";
                }
#else
                // Parse Gemini Response
                if (response_json.contains("candidates")) {
                    std::cout << "--- AUDIT REPORT ---\n" 
                              << response_json["candidates"][0]["content"]["parts"][0]["text"].get<std::string>() 
                              << "\n--------------------\n";
                }
#endif
                else {
                    std::cerr << "[API ERROR] " << response_string << std::endl;
                }
            } catch (...) {
                std::cerr << "[ERROR] Failed to parse JSON." << std::endl;
            }
        }
        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: ./cuda-audit <path_to_kernel.cu>" << std::endl;
        return 1;
    }

#ifdef USE_CLAUDE
    const char* api_key_env = std::getenv("ANTHROPIC_API_KEY");
#else
    const char* api_key_env = std::getenv("GEMINI_API_KEY");
#endif

    if (!api_key_env) {
        std::cerr << "[ERROR] Missing API Key environment variable." << std::endl;
        return 1;
    }

    audit_kernel(read_file(argv[1]), api_key_env);
    return 0;
}

