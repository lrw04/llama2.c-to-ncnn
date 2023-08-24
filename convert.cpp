// Note: the code abuses the property that for the Llama 2 7B model,
// n_heads = n_kv_heads

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

using param = std::variant<float, long long, std::string>;

uint16_t fp32_to_fp16(float f) {
    uint32_t ff;
    memcpy(&ff, &f, sizeof(float));  // avoid UB

    uint32_t sign = ff >> 31 & 1, exponent = ff >> 23 & 0xff,
             significand = ff & 0x7fffff;
    uint16_t fp16;
    if (!exponent) {
        fp16 = (sign << 15) | (0x00 << 10) | 0x000;
    } else if (exponent == 0xff) {
        fp16 = (sign << 15) | (0x1f << 10) | (significand ? 0x200 : 0x000);
    } else {
        int newexp = (int)exponent - 127 + 15;
        if (newexp >= 31) {
            fp16 = (sign << 15) | (0x1f << 10) | 0x000;
        } else if (newexp <= 0) {
            fp16 = (sign << 15) | (0x00 << 10) | 0x000;
        } else {
            fp16 = (sign << 15) | (newexp << 10) | (significand >> 13);
        }
    }
    return fp16;
}

struct config {
    long long dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size,
        ctx_len;
};

struct llama_weights {
    float *weights;
    float *embed, *rms_att, *rms_ffn, *wq, *wk, *wv, *wo, *w1, *w2, *w3,
        *rms_final, *freqs_re, *freqs_im, *unembed;
    void fill_fields(config c);
};

void llama_weights::fill_fields(config c) {
    bool shared = c.vocab_size > 0;
    c.vocab_size = abs(c.vocab_size);
    embed = weights;
    rms_att = embed + c.vocab_size * c.dim;
    wq = rms_att + c.n_layers * c.dim;
    wk = wq + c.n_layers * c.dim * c.dim;
    wv = wk + c.n_layers * c.dim * c.dim;
    wo = wv + c.n_layers * c.dim * c.dim;
    rms_ffn = wo + c.n_layers * c.dim * c.dim;
    w1 = rms_ffn + c.n_layers * c.dim;
    w2 = w1 + c.n_layers * c.dim * c.hidden_dim;
    w3 = w2 + c.n_layers * c.dim * c.hidden_dim;
    rms_final = w3 + c.n_layers * c.dim * c.hidden_dim;
    long long head_size = c.dim / c.n_heads;
    freqs_re = rms_final + c.dim;
    freqs_im = freqs_re + c.ctx_len * head_size / 2;
    unembed = shared ? embed : freqs_im + c.ctx_len * head_size / 2;
}

void read_config(FILE *f, config &conf) {
    int x;
    fread(&x, sizeof(int), 1, f);
    conf.dim = x;
    fread(&x, sizeof(int), 1, f);
    conf.hidden_dim = x;
    fread(&x, sizeof(int), 1, f);
    conf.n_layers = x;
    fread(&x, sizeof(int), 1, f);
    conf.n_heads = x;
    fread(&x, sizeof(int), 1, f);
    conf.n_kv_heads = x;
    fread(&x, sizeof(int), 1, f);
    conf.vocab_size = x;
    fread(&x, sizeof(int), 1, f);
    conf.ctx_len = x;
}

size_t weight_size(config c) {
    bool shared = c.vocab_size > 0;
    c.vocab_size = abs(c.vocab_size);
    size_t size = 0;
    size += (long long)c.vocab_size * c.dim;                   // embedding
    size += (long long)c.n_layers * c.dim * 2;                 // rms norms
    size += (long long)c.n_layers * c.dim * c.dim * 4;         // wq, wk, wv, wo
    size += (long long)c.n_layers * c.hidden_dim * c.dim * 3;  // w1, w2, w3
    size += c.dim;                                             // final rms norm
    size += (long long)c.ctx_len * (c.dim / c.n_heads);        // freqs
    if (!shared) size += (long long)c.vocab_size * c.dim;      // unembedding
    return size;
}

struct op {
    std::string type, desc;
    float *weights;
    size_t weight_size;
    std::vector<int> inputs, outputs;
    std::map<int, param> params;
    bool write_flag;
    op() {
        weights = nullptr;
        weight_size = 0;
        write_flag = false;
        desc = "";
    }
};

struct graph {
    std::vector<op> ops;
    int cnt_operands;
    graph() { cnt_operands = 0; }
    void write(std::string name, int global_input, int global_output) {
        std::ofstream param(name + ".param"),
            bin(name + ".bin", std::ios::binary);
        std::vector<int> outd(cnt_operands);
        param << std::fixed << std::setprecision(6);
        for (auto op : ops) {
            for (auto i : op.inputs) outd[i]++;
        }
        std::unordered_map<int, std::vector<int>> splits;
        int new_operands = cnt_operands;
        for (int i = 0; i < cnt_operands; i++) {
            if (outd[i] > 1) {
                for (int j = 0; j < outd[i]; j++)
                    splits[i].push_back(new_operands++);
            }
        }
        std::vector<op> new_ops;
        for (auto o : ops) {
            new_ops.push_back(o);
            for (auto &in : new_ops.back().inputs) {
                if (splits.count(in)) {
                    int orig = in;
                    in = splits[in].back();
                    splits[orig].pop_back();
                }
            }
            for (auto &out : o.outputs) {
                if (outd[out] > 1) {
                    new_ops.emplace_back();
                    op &o = new_ops.back();
                    o.type = "Split";
                    o.inputs.push_back(out);
                    for (auto split : splits[out]) o.outputs.push_back(split);
                }
            }
        }

        long bytes = 0;
        param << "7767517\n";
        param << new_ops.size() << " " << new_operands << "\n";
        for (int i = 0; i < (int)new_ops.size(); i++) {
            const op &o = new_ops[i];
            param << o.type;
            param << " " << o.desc << "_op" << i;
            param << " " << o.inputs.size() << " " << o.outputs.size();
            for (auto in : o.inputs) {
                if (in == global_input)
                    param << " in";
                else
                    param << " " << in;
            }
            for (auto out : o.outputs) {
                if (out == global_input)
                    param << " in";
                else if (out == global_output)
                    param << " out";
                else
                    param << " " << out;
            }
            for (auto [k, v] : o.params) {
                param << " " << k << "=";
                switch (v.index()) {
                    case 0:
                        param << std::get<0>(v);
                        break;
                    case 1:
                        param << std::get<1>(v);
                        break;
                    case 2:
                        param << std::get<2>(v);
                        break;
                }
            }
            param << "\n";

            if (o.weights) {
                // std::cerr << o.desc << " " << o.weight_size << std::endl;
                // for (int i = 0; i < 4; i++) std::cerr << o.weights[i] << " ";
                // std::cerr << std::endl;
                if (o.write_flag) {
                    char f16_magic[] = {0x47, 0x6b, 0x30, 0x01};  // 0x01306B47
                    bin.write(f16_magic, 4);
                    bytes += 4;
                    for (size_t i = 0; i < o.weight_size; i++) {
                        char f16_bytes[2];
                        uint16_t f16 = fp32_to_fp16(o.weights[i]);
                        memcpy(f16_bytes, &f16, 2);
                        bin.write(f16_bytes, 2);
                        bytes += 2;
                    }
                    // char f32_magic[] = {0x00, 0x00, 0x00, 0x00};
                    // bin.write(f32_magic, 4);
                    // bytes += 4;
                    // for (size_t i = 0; i < o.weight_size; i++) {
                    //     char f32_bytes[4];
                    //     memcpy(f32_bytes, o.weights + i, 4);
                    //     bin.write(f32_bytes, 4);
                    //     bytes += 4;
                    // }
                } else {
                    for (size_t i = 0; i < o.weight_size; i++) {
                        char f32_bytes[4];
                        memcpy(f32_bytes, o.weights + i, 4);
                        bin.write(f32_bytes, 4);
                        bytes += 4;
                    }
                }
                char zero = 0;
                while (bytes % 4) {
                    bin.write(&zero, 1);
                    bytes++;
                }
            }
        }
    }
    int input(std::string desc = "") {
        ops.emplace_back();
        ops.back().type = "Input";
        ops.back().desc = desc;
        ops.back().outputs.push_back(cnt_operands);
        return cnt_operands++;
    }
    int constant(float *wt, int w, int h, int c, std::string desc = "") {
        ops.emplace_back();
        op &o = ops.back();
        o.type = "MemoryData";
        o.desc = desc;
        o.params[0] = w;
        o.params[1] = h;
        o.params[2] = c;
        o.weights = wt;
        o.weight_size = (w ? w : 1) * (h ? h : 1) * (c ? c : 1);
        o.outputs.push_back(cnt_operands);
        return cnt_operands++;
    }
    int add(int a, int b, std::string desc = "") {
        ops.emplace_back();
        op &o = ops.back();
        o.type = "BinaryOp";
        o.desc = desc;
        o.params[0] = 0;  // add
        o.inputs.push_back(a);
        o.inputs.push_back(b);
        o.outputs.push_back(cnt_operands);
        return cnt_operands++;
    }
    int minus(int a, int b, std::string desc = "") {
        ops.emplace_back();
        op &o = ops.back();
        o.type = "BinaryOp";
        o.desc = desc;
        o.params[0] = 1;  // minus
        o.inputs.push_back(a);
        o.inputs.push_back(b);
        o.outputs.push_back(cnt_operands);
        return cnt_operands++;
    }
    int addall(int a, float b, std::string desc = "") {
        ops.emplace_back();
        op &o = ops.back();
        o.type = "BinaryOp";
        o.desc = desc;
        o.params[0] = 0;  // add
        o.inputs.push_back(a);
        o.params[1] = 1;
        o.params[2] = b;
        o.outputs.push_back(cnt_operands);
        return cnt_operands++;
    }
    int div(int a, float b, std::string desc = "") {
        ops.emplace_back();
        op &o = ops.back();
        o.type = "BinaryOp";
        o.desc = desc;
        o.params[0] = 3;  // div
        o.inputs.push_back(a);
        o.params[1] = 1;
        o.params[2] = b;
        o.outputs.push_back(cnt_operands);
        return cnt_operands++;
    }
    int elemwisemul(int a, int b, std::string desc = "") {
        ops.emplace_back();
        op &o = ops.back();
        o.type = "BinaryOp";
        o.desc = desc;
        o.params[0] = 2;
        o.inputs.push_back(a);
        o.inputs.push_back(b);
        o.outputs.push_back(cnt_operands);
        return cnt_operands++;
    }
    int rsqrt(int a, std::string desc = "") {
        ops.emplace_back();
        op &o = ops.back();
        o.type = "UnaryOp";
        o.desc = desc;
        o.params[0] = 6;
        o.inputs.push_back(a);
        o.outputs.push_back(cnt_operands);
        return cnt_operands++;
    }
    int mean_1_keepdim(int a, std::string desc = "") {
        ops.emplace_back();
        op &o = ops.back();
        o.type = "Reduction";
        o.desc = desc;
        o.params[0] = 3;
        o.params[1] = 0;
        o.params[4] = 1;
        o.params[5] = 1;
        o.params[-23303] = "1,1";
        o.inputs.push_back(a);
        o.outputs.push_back(cnt_operands);
        return cnt_operands++;
    }
    int square(int a, std::string desc = "") {
        ops.emplace_back();
        op &o = ops.back();
        o.type = "UnaryOp";
        o.desc = desc;
        o.params[0] = 4;
        o.inputs.push_back(a);
        o.outputs.push_back(cnt_operands);
        return cnt_operands++;
    }
    int embed(llama_weights w, config c, int in, std::string desc = "") {
        ops.emplace_back();
        op &o = ops.back();
        o.type = "Embed";
        o.desc = desc;
        o.write_flag = true;
        o.params[0] = c.dim;
        o.params[1] = c.vocab_size;
        o.params[2] = 0;
        o.params[3] = c.dim * c.vocab_size;
        o.inputs.push_back(in);
        o.outputs.push_back(cnt_operands);
        o.weight_size = c.dim * c.vocab_size;
        o.weights = w.embed;
        return cnt_operands++;
    }
    int linear(int a, int in_batch, long long infeat, long long outfeat,
               float *weights, std::string desc = "") {
        ops.emplace_back();
        op &o = ops.back();
        o.type = "Gemm";
        o.desc = desc;
        o.write_flag = true;
        o.params[2] = 0;
        o.params[3] = 1;
        o.params[4] = 0;
        o.params[5] = 1;
        o.params[6] = 1;
        o.params[7] = in_batch;
        o.params[8] = outfeat;
        o.params[9] = infeat;
        o.params[10] = -1;
        o.inputs.push_back(a);
        o.outputs.push_back(cnt_operands);
        o.weights = weights;
        o.weight_size = infeat * outfeat;
        return cnt_operands++;
    }
    int linear_cis(int a, int in_batch, long long infeat, long long outfeat,
               float *weights, std::string desc = "") {
        ops.emplace_back();
        op &o = ops.back();
        o.type = "Gemm";
        o.desc = desc;
        o.write_flag = true;
        o.params[2] = 0;
        o.params[3] = 0;
        o.params[4] = 0;
        o.params[5] = 1;
        o.params[6] = 1;
        o.params[7] = in_batch;
        o.params[8] = outfeat;
        o.params[9] = infeat;
        o.params[10] = -1;
        o.inputs.push_back(a);
        o.outputs.push_back(cnt_operands);
        o.weights = weights;
        o.weight_size = infeat * outfeat;
        return cnt_operands++;
    }
    int matmul(int a, int b, std::string desc = "") {
        ops.emplace_back();
        op &o = ops.back();
        o.type = "MatMul";
        o.desc = desc;
        o.inputs.push_back(a);
        o.inputs.push_back(b);
        o.outputs.push_back(cnt_operands);
        return cnt_operands++;
    }
    std::pair<int, int> unbind2(int a, std::string desc = "") {
        ops.emplace_back();
        op &o = ops.back();
        o.type = "Slice";
        o.desc = desc;
        o.params[-23300] = "2,-233,-233";
        o.params[1] = 3;
        o.inputs.push_back(a);
        int x, y;
        x = cnt_operands++;
        y = cnt_operands++;
        o.outputs.push_back(x);
        o.outputs.push_back(y);
        return {x, y};
    }
    int view4(int a, long long x, long long y, long long z, long long w,
              std::string desc = "") {
        ops.emplace_back();
        op &o = ops.back();
        o.type = "Reshape";
        o.desc = desc;
        o.params[0] = w;
        o.params[1] = z;
        o.params[11] = y;
        o.params[2] = x;
        o.inputs.push_back(a);
        o.outputs.push_back(cnt_operands);
        return cnt_operands++;
    }
    int view3(int a, long long x, long long y, long long z,
              std::string desc = "") {
        ops.emplace_back();
        op &o = ops.back();
        o.type = "Reshape";
        o.desc = desc;
        o.params[0] = z;
        o.params[1] = y;
        o.params[2] = x;
        o.inputs.push_back(a);
        o.outputs.push_back(cnt_operands);
        return cnt_operands++;
    }
    int view2(int a, long long x, long long y, std::string desc = "") {
        ops.emplace_back();
        op &o = ops.back();
        o.type = "Reshape";
        o.desc = desc;
        o.params[0] = y;
        o.params[1] = x;
        o.inputs.push_back(a);
        o.outputs.push_back(cnt_operands);
        return cnt_operands++;
    }
    int stack(int a, int b, int axis, std::string desc = "") {
        ops.emplace_back();
        op &o = ops.back();
        o.type = "Concat";
        o.desc = desc;
        o.params[0] = axis;
        o.inputs.push_back(a);
        o.inputs.push_back(b);
        o.outputs.push_back(cnt_operands);
        return cnt_operands++;
    }
    int transpose3_01(int a, std::string desc = "") {
        ops.emplace_back();
        op &o = ops.back();
        o.type = "Permute";
        o.desc = desc;
        o.params[0] = 2;
        o.inputs.push_back(a);
        o.outputs.push_back(cnt_operands);
        return cnt_operands++;
    }
    int transpose3_12(int a, std::string desc = "") {
        ops.emplace_back();
        op &o = ops.back();
        o.type = "Permute";
        o.desc = desc;
        o.params[0] = 1;
        o.inputs.push_back(a);
        o.outputs.push_back(cnt_operands);
        return cnt_operands++;
    }
    int softmax_dim_1(int a, std::string desc = "") {
        ops.emplace_back();
        op &o = ops.back();
        o.type = "Softmax";
        o.desc = desc;
        o.params[0] = 2;
        o.params[1] = 1;
        o.inputs.push_back(a);
        o.outputs.push_back(cnt_operands);
        return cnt_operands++;
    }
    int silu(int a, std::string desc = "") {
        ops.emplace_back();
        op &o = ops.back();
        o.type = "Swish";
        o.desc = desc;
        o.inputs.push_back(a);
        o.outputs.push_back(cnt_operands);
        return cnt_operands++;
    }
    std::pair<int, int> apply_rotary_emb(int xq, int xk, config c,
                                         int freqs_cos, int freqs_sin) {
        xq = view4(xq, c.ctx_len, c.n_heads, c.dim / c.n_heads / 2, 2,
                   "xq_view");
        xk = view4(xk, c.ctx_len, c.n_heads, c.dim / c.n_heads / 2, 2,
                   "xk_view");
        auto [xq_r, xq_i] = unbind2(xq, "xq_unbind");
        auto [xk_r, xk_i] = unbind2(xk, "xk_unbind");
        xq_r = view3(xq_r, c.ctx_len, c.n_heads, c.dim / c.n_heads / 2, "xq_r");
        xq_i = view3(xq_i, c.ctx_len, c.n_heads, c.dim / c.n_heads / 2, "xq_i");
        xk_r = view3(xk_r, c.ctx_len, c.n_heads, c.dim / c.n_heads / 2, "xk_r");
        xk_i = view3(xk_i, c.ctx_len, c.n_heads, c.dim / c.n_heads / 2, "xk_i");
        auto xq_out_r = minus(elemwisemul(xq_r, freqs_cos),
                              elemwisemul(xq_i, freqs_sin), "xq_out_r");
        auto xq_out_i = add(elemwisemul(xq_r, freqs_sin),
                            elemwisemul(xq_i, freqs_cos), "xq_out_i");
        auto xk_out_r = minus(elemwisemul(xk_r, freqs_cos),
                              elemwisemul(xk_i, freqs_sin), "xk_out_r");
        auto xk_out_i = add(elemwisemul(xk_r, freqs_sin),
                            elemwisemul(xk_i, freqs_cos), "xk_out_i");

        auto xq_out = view3(stack(xq_out_r, xq_out_i, 2), c.ctx_len, c.n_heads,
                            c.dim / c.n_heads, "xq_out");
        auto xk_out = view3(stack(xk_out_r, xk_out_i, 2), c.ctx_len, c.n_heads,
                            c.dim / c.n_heads, "xk_out");

        return {xq_out, xk_out};
    }
    int rmsnorm_att(int layer, llama_weights w, config c, int in,
                    float eps = 1e-5f) {
        float *weight = w.rms_att + layer * c.dim;
        return elemwisemul(
            elemwisemul(in, rsqrt(addall(mean_1_keepdim(square(in)), eps)),
                        "norm"),
            constant(weight, c.dim, 1, 0, "gain_att_" + std::to_string(layer)),
            "gainmul_att_" + std::to_string(layer));
    }
    int rmsnorm_ffn(int layer, llama_weights w, config c, int in,
                    float eps = 1e-5f) {
        float *weight = w.rms_ffn + layer * c.dim;
        return elemwisemul(
            elemwisemul(in, rsqrt(addall(mean_1_keepdim(square(in)), eps))),
            constant(weight, c.dim, 1, 0, "gain_ffn_" + std::to_string(layer)),
            "gainmul_ffn_" + std::to_string(layer));
    }
    int rmsnorm_final(llama_weights w, config c, int in, float eps = 1e-5f) {
        float *weight = w.rms_final;
        return elemwisemul(
            elemwisemul(in, rsqrt(addall(mean_1_keepdim(square(in)), eps))),
            constant(weight, c.dim, 1, 0, "finalgain"), "gainmul_final");
    }
    int attention(int layer, llama_weights w, config c, int in, int freqs_cos,
                  int freqs_sin, int mask) {
        int head_dim = c.dim / c.n_heads;
        auto xq = view3(
            linear(in, c.ctx_len, c.dim, c.dim, w.wq + layer * c.dim * c.dim,
                   "xq_linear" + std::to_string(layer)),
            c.ctx_len, c.n_heads, c.dim / c.n_heads,
            "xq" + std::to_string(layer));
        auto xk = view3(
            linear(in, c.ctx_len, c.dim, c.dim, w.wk + layer * c.dim * c.dim,
                   "xk_linear" + std::to_string(layer)),
            c.ctx_len, c.n_heads, c.dim / c.n_heads,
            "xk" + std::to_string(layer));
        auto xv = view3(
            linear(in, c.ctx_len, c.dim, c.dim, w.wv + layer * c.dim * c.dim,
                   "xv_linear" + std::to_string(layer)),
            c.ctx_len, c.n_heads, c.dim / c.n_heads,
            "xv" + std::to_string(layer));
        auto [xqr, xkr] = apply_rotary_emb(xq, xk, c, freqs_cos, freqs_sin);
        xq = transpose3_01(xqr, "xq_01_" + std::to_string(layer));
        xk = transpose3_01(xkr, "xk_01_" + std::to_string(layer));
        xv = transpose3_01(xv, "xv_01_" + std::to_string(layer));
        auto scores =
            div(matmul(xq, transpose3_12(xk, "xk_12"),
                       "scores_matmul_" + std::to_string(layer)),
                sqrt(c.dim / c.n_heads), "scores_div_" + std::to_string(layer));
        scores = add(scores, mask, "addmask_" + std::to_string(layer));
        scores =
            softmax_dim_1(scores, "scores_softmax_" + std::to_string(layer));
        auto out = matmul(scores, xv);
        out = transpose3_01(out);
        out = view2(out, c.ctx_len, c.dim);
        return linear(out, c.ctx_len, c.dim, c.dim,
                      w.wo + layer * c.dim * c.dim,
                      "out_linear_" + std::to_string(layer));
    }
    int ffn(int layer, llama_weights w, config c, int in) {
        return linear(
            elemwisemul(silu(linear(in, c.ctx_len, c.dim, c.hidden_dim,
                                    w.w1 + layer * c.dim * c.hidden_dim,
                                    "w1_" + std::to_string(layer))),
                        linear(in, c.ctx_len, c.dim, c.hidden_dim,
                               w.w3 + layer * c.dim * c.hidden_dim,
                               "w3_" + std::to_string(layer))),
            c.ctx_len, c.hidden_dim, c.dim, w.w2 + layer * c.hidden_dim * c.dim,
            "w2_" + std::to_string(layer));
    }
    int transformer_block(int layer, llama_weights w, config c, int in,
                          int freqs_cos, int freqs_sin, int mask) {
        auto h = add(in,
                     attention(layer, w, c, rmsnorm_att(layer, w, c, in),
                               freqs_cos, freqs_sin, mask),
                     "tb_h_" + std::to_string(layer));
        auto out = add(h, ffn(layer, w, c, rmsnorm_ffn(layer, w, c, h)),
                       "layer_" + std::to_string(layer));
        return out;
    }
};

// cvtbin llama-2-7b.bin model
// -> model.param, model.bin
int main(int argc, char **argv) {
    std::string bin_path, model;
    if (argc == 3) {
        bin_path = argv[1];
        model = argv[2];
    } else {
        bin_path = "7b.bin";
        model = "7b.ncnn";
    }

    FILE *bin = fopen(bin_path.c_str(), "rb");
    config conf;
    llama_weights weights;
    read_config(bin, conf);
    size_t size = weight_size(conf);
    weights.weights = (float *)malloc(sizeof(float) * size);
    fread(weights.weights, sizeof(float), size, bin);
    weights.fill_fields(conf);
    fclose(bin);
    bin = nullptr;

    conf.vocab_size = abs(conf.vocab_size);

    std::vector<float> mask_v(conf.ctx_len * conf.ctx_len);
    for (size_t i = 0; i < conf.ctx_len; i++)
        for (size_t j = 0; j < conf.ctx_len; j++)
            mask_v[i * conf.ctx_len + j] = j > i ? -1.0f / 0.0f : 0;

    graph g;
    int input = g.input("input");
    int x = g.embed(weights, conf, input, "embed");
    int freqs_cos = g.constant(weights.freqs_re, conf.dim / conf.n_heads / 2, 1,
                               conf.ctx_len, "freqs_cos");
    int freqs_sin = g.constant(weights.freqs_im, conf.dim / conf.n_heads / 2, 1,
                               conf.ctx_len, "freqs_sin");
    int mask = g.constant(mask_v.data(), conf.ctx_len, conf.ctx_len, 1, "mask");
    for (int i = 0; i < conf.n_layers; i++)
        x = g.transformer_block(i, weights, conf, x, freqs_cos, freqs_sin,
                                mask);
    x = g.rmsnorm_final(weights, conf, x);
    x = g.linear(x, conf.ctx_len, conf.dim, conf.vocab_size, weights.unembed,
                 "unembed");

    g.write(model, input, x);
    std::ofstream desc(model + ".desc");
    desc << conf.ctx_len << std::endl;
}
