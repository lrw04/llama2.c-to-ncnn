# llama2.c-to-ncnn

A converter for llama2.c legacy models to ncnn models. Currently, this is only tested on the 7B model.

## Compiling

Set the NCNN_DIR directory to your directory for your ncnn source tree, with a build.

```
make
```

You will get two binaries in the current folder, `convert` and `inference`. `convert` is the converter from llama2.c legacy format to ncnn format, and `inference` is an example of how to use the resulting ncnn models.

## Converting Meta's weights

Use `export.py` from <https://github.com/karpathy/llama2.c>.

## Converting weights into ncnn's format

```
./convert 7b.bin 7b.ncnn
```

You will get `7b.ncnn.bin`, `7b.ncnn.param` and `7b.ncnn.desc`. For any model with the three files, the common name `7b.ncnn` is used to denote the model.

## Complete text using the resulting model

```
./inference <MODEL> <PROMPT> <N-TOKENS>
```

## Example outputs

> Iwasawa theory is an isomorphism of the following four categories: (i) Category of $k$-rational points of $\mathcal{G}$ and (ii) Category of $k$-rational points of $\mathcal{G}/\mathcal{K}_H$, where $\mathcal{K}_H$ denotes the relative kernel of the Hilbert modular abelian scheme $\mathcal{H}$, (iii)

Generated with the 7B model. <del>Typical nonsense generated by a small LLM</del>

## TODO

- [X] KV cache
- [ ] Chat completion support
- [ ] int8 quantization
- [ ] Reduce memory usage during conversion
