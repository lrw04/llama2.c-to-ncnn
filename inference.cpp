#include <algorithm>
#include <cassert>
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

const float temp = 1, topp = 0.9;
const int topk = 300;

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
    if (s.length() && s[0] != ' ') s = " " + s;
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
    std::vector<ncnn::Mat> kcache, vcache, kcp, vcp;
    int ctx_length, pos, n_l, dim, n_heads;
    std::vector<float> freqs_cos, freqs_sin;
    tinyllama(std::string bin, std::string param, int n_layers, int ctx_len,
              int dim_, int nh);
    std::vector<float> forward(int token);
};

tinyllama::tinyllama(std::string bin, std::string param, int n_layers,
                     int ctx_len, int dim_, int nh) {
    if (net.load_param(param.c_str())) exit(1);
    if (net.load_model(bin.c_str())) exit(1);
    pos = 0;
    n_l = n_layers;
    ctx_length = ctx_len;
    dim = dim_;
    n_heads = nh;
    kcache.resize(n_l);
    vcache.resize(n_l);
    kcp.resize(n_l);
    vcp.resize(n_l);
    int head_dim = dim / n_heads;
    freqs_cos.resize(ctx_length * head_dim / 2);
    freqs_sin.resize(ctx_length * head_dim / 2);

    for (int i = 0; i < ctx_length; i++) {
        for (int j = 0; j < head_dim / 2; j++) {
            auto x = i / pow(10000.0, j * 2 / (double)head_dim);
            freqs_cos[i * head_dim / 2 + j] = cos(x);
            freqs_sin[i * head_dim / 2 + j] = sin(x);
        }
    }

    for (int i = 0; i < n_l; i++) {
        kcache[i].create(dim, 0);
        vcache[i].create(dim, 0);
    }
}

std::vector<float> tinyllama::forward(int token) {
    ncnn::Mat x(1), fc, fs;
    *((int*)x) = token;

    int head_dim = dim / n_heads;
    fc.create(head_dim / 2, pos + 1);
    fs.create(head_dim / 2, pos + 1);
    for (int i = 0; i < (pos + 1) * head_dim / 2; i++) {
        fc[i] = freqs_cos[i];
        fs[i] = freqs_sin[i];
    }

    auto ex = net.create_extractor();
    ex.input("in", x);
    ex.input("freqs_cos", fc);
    ex.input("freqs_sin", fs);
    for (int i = 0; i < n_l; i++) {
        auto layer_name = std::to_string(i);
        auto kc_name = "kcache." + layer_name;
        auto vc_name = "vcache." + layer_name;
        ex.input(kc_name.c_str(), kcache[i]);
        ex.input(vc_name.c_str(), vcache[i]);
    }
    ncnn::Mat logits_mat;
    for (int i = 0; i < n_l; i++) {
        auto layer_name = std::to_string(i);
        auto kc_name = "kcache_out." + layer_name;
        auto vc_name = "vcache_out." + layer_name;
        ex.extract(kc_name.c_str(), kcp[i]);
        ex.extract(vc_name.c_str(), vcp[i]);
    }
    ex.extract("out", logits_mat);
    std::vector<float> logits(logits_mat.total());

    for (int i = 0; i < n_l; i++) {
        kcache[i] = kcp[i].clone();
        vcache[i] = vcp[i].clone();
    }

    for (size_t i = 0; i < logits_mat.total(); i++) logits[i] = logits_mat[i];

    pos++;
    if (pos == ctx_length) {
        pos--;
        auto shift_cache = [&](ncnn::Mat& x) -> void {
            ncnn::Mat y;
            y.create(dim, ctx_length);
            for (int i = 0; i < dim * ctx_length; i++) {
                y[i] = x[i + dim];
            }
            x = y;
        };
        auto shift = [&](std::vector<ncnn::Mat>& v) -> void {
            for (auto& x : v) {
                shift_cache(x);
            }
        };
        shift(kcache);
        shift(vcache);
    }

    return logits;
}

int sample(const std::vector<float>& logits, float temp, float topp, int topk) {
    // return std::max_element(logits.begin(), logits.end()) - logits.begin();

    assert(logits.size() == vocab_size);

    static std::mt19937_64 rng(3407);  // haha
    static std::uniform_real_distribution<float> dist(0, 1);

    std::vector<std::pair<float, int>> probs(vocab_size);
    for (int i = 0; i < vocab_size; i++) probs[i] = {logits[i] / temp, i};
    std::sort(probs.begin(), probs.end(),
              std::greater<std::pair<float, int>>());
    while (probs.size() > topk) probs.pop_back();

    // softmax
    auto maximum = probs[0].first;
    std::transform(probs.begin(), probs.end(), probs.begin(),
                   [maximum](auto x) {
                       return std::make_pair(expf(x.first - maximum), x.second);
                   });
    auto sum = std::accumulate(probs.begin(), probs.end(), 0.0f,
                               [](auto x, auto y) { return x + y.first; });
    std::transform(probs.begin(), probs.end(), probs.begin(), [sum](auto x) {
        return std::make_pair(x.first / sum, x.second);
    });

    sum = 0;
    int last = 0;
    for (int i = 0; i < (int)probs.size(); i++) {
        sum += probs[i].first;
        last = i;
        if (sum > topp) break;
    }

    float r = dist(rng) * sum;
    sum = 0;
    for (int i = 0; i <= last; i++) {
        sum += probs[i].first;
        if (sum > r) return probs[i].second;
    }
    return probs[last].second;
}

// ./inference MODEL PROMPT OUT-TOKEN-COUNT
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

    int ctx_len, n_layers, dim, n_heads;
    std::ifstream desc(std::string(argv[1]) + ".desc");
    desc >> ctx_len >> n_layers >> dim >> n_heads;
    desc.close();

    tinyllama model(model_bin, model_param, n_layers, ctx_len, dim, n_heads);

    // tokenize prompt
    bpe tokenizer;
    tokenizer.load(tokenizer_path);

    auto tokens = tokenizer.encode(prompt);
    tokens.insert(tokens.begin(), 1);  // bos
    int prompt_end = tokens.size();
    tokens.resize(token_count);

    // for (int i = 0; i < token_count; i++) std::cout << tokens[i] << " ";
    // std::cout << std::endl;

    // feed forward
    for (int i = 0; i < token_count; i++) {
        std::cout << tokenizer.vocab[tokens[i]] << std::flush;
        auto logits = model.forward(tokens[i]);
        if (i < prompt_end - 1) continue;
        tokens[i + 1] = sample(logits, temp, topp, topk);
    }
    std::cout << std::endl;
}
