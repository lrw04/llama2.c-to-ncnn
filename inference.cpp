#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "net.h"

const int vocab_size = 32000;

std::mt19937_64 rng(3407);  // qwq
std::uniform_real_distribution<float> dist(0, 1);

struct bpe {
    int max_token_length;
    std::vector<std::string> vocab;
    std::unordered_map<std::string, int> lookup;
    std::vector<float> scores;

    void load(std::string path);
    std::vector<int> encode(std::string s);
};

void bpe::load(std::string path) {
    vocab.resize(vocab_size);
    scores.resize(vocab_size);
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) exit(1);
    fread(&max_token_length, sizeof(int), 1, f);
    std::vector<char> s(max_token_length + 1);
    for (int i = 0; i < vocab_size; i++) {
        fread(scores.data() + i, sizeof(float), 1, f);
        int len;
        fread(&len, sizeof(int), 1, f);
        fread(s.data(), sizeof(char) * len, 1, f);
        s[len] = 0;
        vocab[i] = s.data();
        lookup[vocab[i]] = i;
    }
    fclose(f);
}

std::vector<int> bpe::encode(std::string s) {
    std::vector<int> tokens;
    for (size_t i = 0; i < s.length(); i++) {
        std::string c;
        c += s[i];
        int id = lookup[c];
        tokens.push_back(id);
    }

    while (true) {
        float best_score = -1e10;
        int best_index = -1, best_token = -1;

        for (size_t i = 0; i + 1 < tokens.size(); i++) {
            auto merged = vocab[tokens[i]] + vocab[tokens[i + 1]];
            if (lookup.count(merged) && scores[lookup[merged]] > best_score) {
                best_score = scores[lookup[merged]];
                best_index = i;
                best_token = lookup[merged];
            }
        }

        if (best_token == -1) break;

        tokens[best_index] = best_token;
        tokens.erase(tokens.begin() + best_index + 1);
    }
    return tokens;
}

struct tinyllama {
    ncnn::Net net;
    float topp, temp;
    int topk, ctx_length;
    tinyllama(std::string bin, std::string param);
    int forward(const std::vector<int>& tokens);
};

tinyllama::tinyllama(std::string bin, std::string param) {
    if (net.load_param(param.c_str())) exit(1);
    if (net.load_model(bin.c_str())) exit(1);
}

int tinyllama::forward(const std::vector<int>& tokens) {
    std::vector<int> input(ctx_length, 0);
    int ind;
    if (tokens.size() > ctx_length) {
        for (int i = 0; i < ctx_length; i++)
            input[i] = tokens[tokens.size() - ctx_length + i];
        ind = ctx_length - 1;
    } else {
        for (int i = 0; i < (int)tokens.size(); i++) input[i] = tokens[i];
        ind = (int)tokens.size() - 1;
    }

    ncnn::Mat in(ctx_length);
    for (int i = 0; i < ctx_length; i++) ((int*)in)[i] = input[i];
    auto ex = net.create_extractor();
    ex.input("in", in);
    ncnn::Mat out;
    int ret = ex.extract("out", out);

    // bool nan = false;
    // std::cerr << out.elemsize << std::endl;
    // for (size_t i = 0; i < out.total(); i++)
    //     if (isnan(out[i])) nan = true;
    // std::cerr << "out nan: " << nan << std::endl;

    std::vector<float> logits(vocab_size);
    for (int i = 0; i < vocab_size; i++) logits[i] = out.row(ind)[i];

    for (int i = 0; i < vocab_size; i++) logits[i] /= temp;

    // sampling
    std::vector<std::pair<int, float>> a;
    for (int i = 0; i < vocab_size; i++) a.emplace_back(i, logits[i]);
    std::sort(a.begin(), a.end(),
              [](auto x, auto y) { return x.second > y.second; });
    while (a.size() > topk) a.pop_back();
    auto maximum = std::max_element(a.begin(), a.end(), [](auto x, auto y) {
                       return x.second < y.second;
                   })->second;
    std::transform(a.begin(), a.end(), a.begin(),
                   [maximum](auto x) -> std::pair<int, float> {
                       return {x.first, expf(x.second - maximum)};
                   });
    float sum = std::accumulate(a.begin(), a.end(), 0.0f,
                                [](float x, auto y) { return x + y.second; });
    std::transform(a.begin(), a.end(), a.begin(),
                   [sum](auto x) -> std::pair<int, float> {
                       return {x.first, x.second / sum};
                   });

    sum = 0;
    int last = 0;
    for (int i = 0; i < (int)a.size(); i++) {
        sum += a[i].second;
        last = i;
        if (sum > topp) break;
    }

    float r = dist(rng) * sum;
    sum = 0;
    for (int i = 0; i <= last; i++) {
        sum += a[i].second;
        if (sum > r) {
            // std::cerr << a[i].second;
            return a[i].first;
        }
    }

    // std::cerr << a[last].second;
    return a[last].first;
}

// ./tinyllamas MODEL PROMPT OUT-TOKEN-COUNT
int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " MODEL PROMPT OUT-TOKEN-COUNT"
                  << std::endl;
        return 1;
    }

    std::string model_bin = argv[1], model_param = argv[1],
                tokenizer_path = "tokenizer.bin", prompt = argv[2];
    int token_count = std::stoi(argv[3]);
    model_bin += ".bin";
    model_param += ".param";
    std::ifstream desc(std::string(argv[1]) + ".desc");

    tinyllama model(model_bin, model_param);
    model.topp = 0.9f;
    model.temp = 1.0f;
    model.topk = 300;
    desc >> model.ctx_length;
    desc.close();

    // tokenize prompt
    bpe tokenizer;
    tokenizer.load(tokenizer_path);

    auto tokens = tokenizer.encode(prompt);
    // tokens.insert(tokens.begin(), 1);  // bos

    for (auto token : tokens) std::cout << tokenizer.vocab[token] << std::flush;

    // feed forward
    for (int _ = 0; _ < token_count; _++) {
        // for (auto tok : tokens) std::cerr << tok << " ";
        // std::cerr << std::endl;
        auto next = model.forward(tokens);
        std::cout << tokenizer.vocab[next] << std::flush;
        tokens.push_back(next);
    }
    std::cout << "\n";
}
