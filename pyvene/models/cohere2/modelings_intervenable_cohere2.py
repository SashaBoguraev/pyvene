"""
Each modeling file in this library is a mapping between
abstract naming of intervention anchor points and actual
model module defined in the huggingface library.

We also want to let the intervention library know how to
config the dimensions of intervention based on model config
defined in the huggingface library.
"""

import torch
from ..constants import *


cohere2_type_to_module_mapping = {
    "block_input": ("layers[%s]", CONST_INPUT_HOOK),
    "block_output": ("layers[%s]", CONST_OUTPUT_HOOK),
    "mlp_output": ("layers[%s].mlp", CONST_OUTPUT_HOOK),
    "mlp_input": ("layers[%s].mlp", CONST_INPUT_HOOK),
    "attention_value_output": ("layers[%s].self_attn.o_proj", CONST_INPUT_HOOK),
    "head_attention_value_output": ("layers[%s].self_attn.o_proj", CONST_INPUT_HOOK, (split_head_and_permute, "n_head")),
    "attention_output": ("layers[%s].self_attn", CONST_OUTPUT_HOOK),
    "attention_input": ("layers[%s].self_attn", CONST_INPUT_HOOK),
    "query_output": ("layers[%s].self_attn.q_proj", CONST_OUTPUT_HOOK),
    "key_output": ("layers[%s].self_attn.k_proj", CONST_OUTPUT_HOOK),
    "value_output": ("layers[%s].self_attn.v_proj", CONST_OUTPUT_HOOK),
    "head_query_output": ("layers[%s].self_attn.q_proj", CONST_OUTPUT_HOOK, (split_head_and_permute, "n_head")),
    "head_key_output": ("layers[%s].self_attn.k_proj", CONST_OUTPUT_HOOK, (split_head_and_permute, "n_kv_head")),
    "head_value_output": ("layers[%s].self_attn.v_proj", CONST_OUTPUT_HOOK, (split_head_and_permute, "n_kv_head")),
}


cohere2_type_to_dimension_mapping = {
    "n_head": ("num_attention_heads",),
    "n_kv_head": ("num_key_value_heads",),
    "block_input": ("hidden_size",),
    "block_output": ("hidden_size",),
    "mlp_output": ("hidden_size",),
    "mlp_input": ("hidden_size",),
    "attention_value_output": ("hidden_size",),
    "head_attention_value_output": ("hidden_size/num_attention_heads",),
    "attention_output": ("hidden_size",),
    "attention_input": ("hidden_size",),
    "query_output": ("hidden_size",),
    "key_output": ("hidden_size",),
    "value_output": ("hidden_size",),
    "head_query_output": ("hidden_size/num_attention_heads",),
    "head_key_output": ("hidden_size/num_attention_heads",),
    "head_value_output": ("hidden_size/num_attention_heads",),
}


"""Cohere2 model with LM head"""
cohere2_lm_type_to_module_mapping = {}
for k, v in cohere2_type_to_module_mapping.items():
    cohere2_lm_type_to_module_mapping[k] = (f"model.{v[0]}",) + v[1:]


cohere2_lm_type_to_dimension_mapping = cohere2_type_to_dimension_mapping
