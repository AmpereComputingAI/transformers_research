from typing import Optional, Tuple

import torch

from ..utils import logging


logger = logging.get_logger(__name__)


def repeat_kv(tracer, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    tracer.add_op("torch.Tensor.size", {"input": hidden_states},
                  {f"{i}": val for i, val in enumerate([batch, num_key_value_heads, slen, head_dim])})
    if n_rep == 1:
        return hidden_states
    hidden_states_ = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    input_dict = {f"{i}": val for i, val in enumerate([batch, num_key_value_heads, n_rep, slen, head_dim])}
    input_dict["input"] = hidden_states[:, :, None, :, :]
    tracer.add_op("torch.Tensor.expand", input_dict, {"output": hidden_states_})
    hidden_states = hidden_states_
    x = hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
    input_dict = {f"{i}": val for i, val in enumerate([batch, num_key_value_heads * n_rep, slen, head_dim])}
    input_dict["input"] = hidden_states
    tracer.add_op("torch.Tensor.reshape", input_dict, {"output": x})
    return x


def sdpa_attention_forward(
    tracer,
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    is_causal: Optional[bool] = None,
    **kwargs,
) -> Tuple[torch.Tensor, None]:
    if kwargs.get("output_attentions", False) or kwargs.get("head_mask", None) is not None:
        logger.warning_once(
            "`sdpa` attention does not support `output_attentions=True` or `head_mask`."
            " Please set your attention to `eager` if you want any of these features."
        )

    if hasattr(module, "num_key_value_groups"):
        key = repeat_kv(tracer, key, module.num_key_value_groups)
        value = repeat_kv(tracer, value, module.num_key_value_groups)

    if attention_mask is not None and attention_mask.ndim == 4:
        attention_mask = attention_mask[:, :, :, : key.shape[-2]]
        tracer.add_op("torch.Tensor.size", {"input": key}, {"output": key.shape})

    # SDPA with memory-efficient backend is bugged with non-contiguous inputs and custom attn_mask for some torch versions
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
    # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
    # Note that it is important to check first for the shape, otherwise compile will fail with `argument 'is_causal' must be bool, not SymBool`
    if is_causal is None:
        # The last condition is for encoder (decoder) models which specify this by passing their own `is_causal` flag
        # This is mainly due to those models having mixed implementations for encoder, decoder, and encoder-decoder attns
        is_causal = query.shape[2] > 1 and attention_mask is None and getattr(module, "is_causal", True)
        tracer.add_op("torch.Tensor.size", {"input": query}, {"output": query.shape})

    # Shapes (e.g. query.shape[2]) are tensors during jit tracing, resulting in `is_causal` being a tensor.
    # We convert it to a bool for the SDPA kernel that only accepts bools.
    if torch.jit.is_tracing() and isinstance(is_causal, torch.Tensor):
        is_causal = is_causal.item()

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=dropout,
        scale=scaling,
        is_causal=is_causal,
    )
    tracer.add_op("torch.nn.functional.scaled_dot_product_attention",
                  {"query": query, "key": key, "value": value, "attn_mask": attention_mask, "dropout_p": dropout,
                   "scale": scaling, "is_causal": is_causal},
                  {"output": attn_output})

    attn_output_ = attn_output.transpose(1, 2).contiguous()
    tracer.add_op("torch.Tensor.transpose",
                  {"dim0": 1, "dim1": 2},
                  {"output": attn_output_})
    attn_output = attn_output_

    return attn_output, None
