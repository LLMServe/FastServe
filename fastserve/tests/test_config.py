from fastserve.config import ModelConfig, ParallelConfig


def test_model_config():
    model_config = ModelConfig(model="facebook/opt-1.3b", tokenizer="facebook/opt-1.3b")
    assert model_config.get_hidden_size() == 2048
    assert model_config.get_head_size() == 64
    assert model_config.get_num_heads(parallel_config=ParallelConfig(1, 1)) == 32
    assert model_config.get_num_heads(parallel_config=ParallelConfig(2, 1)) == 16
    assert model_config.get_max_model_len() == 2048
    assert model_config.get_num_layers(parallel_config=ParallelConfig(1, 1)) == 24
    assert model_config.get_num_layers(parallel_config=ParallelConfig(1, 0, 2, 1)) == 12
    print(model_config.hf_config)
