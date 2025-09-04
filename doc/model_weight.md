# How to convert PyTorch weights to SwiftTransformer weights

## GPT/OPT

``` Python
python convert-opt.py \
        --input ../opt-weights/opt-125m.pt \
        --output ../opt-weights/opt-125m-fp16 \
        --dtype fp16
# Or, multiple files in glob pattern:
python convert-opt.py \
--input ../opt-weights/opt-30B/reshard-model_part-*.pt \
--output ../opt-weights/opt-30B-fp16 \
--dtype fp16
```


# How SwiftTransformer stores weights

## GPT/OPT

To enable tensor and pipeline parallelism, we split the weights of GPT/OPT into multiple partitions. Each partition is stored in a separate file. Their filenames will be used as keys for weight loading.

There are two kinds of weights:

- General weights
- Layer weights

### General weights

All the GPUs will load the same files for general weights. General weights include:

- decoder.embed_tokens.weight                           ([vocab_size, hidden_size])
- decoder.embed_positions.weight                        ([max_position_embeddings, hidden_size])
- decoder.layer_norm.weight                             ([hidden_size])
- decoder.layer_norm.bias                               ([hidden_size])

### Layer weights

Layer weights are usually split into 8 tensor parallel partitions. Each GPU will load some of them. We classify layer weights into three categories:

#### Divided by dim 0

- decoder.layers.{layer_index}.fc1.weight.tp_{tp_index} ([ffn_dim, hidden_size])
- decoder.layers.{layer_index}.fc1.bias.tp_{tp_index}   ([ffn_dim])
- decoder.layers.{layer_index}.self_attn.out_proj.weight        ([num_local_heads, head_dim, hidden_size])

#### Divided by dim 1

- decoder.layers.{layer_index}.fc2.weight.tp_{tp_index} ([hidden_size, ffn_dim])
- decoder.layers.{layer_index}.self_attn.qkv_proj.bias  ([3, num_local_heads, head_dim])


#### Divided by dim 2

- decoder.layers.{layer_index}.self_attn.qkv_proj.weight ([hidden_size, 3, num_attention_heads, head_dim])

#### Not divided

- decoder.layers.0.fc2.bias                                     ([hidden_size])
- decoder.layers.{layer_index}.self_attn.out_proj.bias          ([hidden_size])
- decoder.layers.{layer_index}.self_attn_layer_norm.weight      ([hidden_size])
- decoder.layers.{layer_index}.self_attn_layer_norm.bias        ([hidden_size])
