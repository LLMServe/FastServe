# Dev notice

## Common issues: C++

1. Use namespace `st` for all the functions and variables.
2. If one function returns status code, it should be checked. Some macro like `CHECK_STATUS` can be used to check the status code and print the error message.
3. Repacakge of external functions should hava a name of `stLibFunc`. For example, `ncclAllReduce` should be repackaged as `stNcclAllReduce`.
4. Please pay attention to that, in our attention layer, we multiple input_tokens with matrix
qkv_weight_kernel instead of qkv_weight_kernel^T. This is different from the common practice
of using `torch.nn.Linear` which multiplys input_tokens with qkv_weight_kernel^T. Please
pay attention to this when converting weights.
