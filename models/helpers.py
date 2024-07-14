"""
This function can be found in torch.nn.Functional.scaled_dot_product_attention if torch version >= 2.0.X.

Due to my conda virtual environment (torch==1.13.0+cu118), I wrote this function to compute scaled dot product attention \
for using build attention blocks in build_models.py(line 440).
"""
import torch
import math
from typing import Optional


def scaled_dot_product_attention(query: torch.Tensor,
                                 key: torch.Tensor,
                                 value: torch.Tensor,
                                 attn_mask: Optional[torch.Tensor] = None,
                                 dropout_p: float = 0.0,
                                 is_causal=False,
                                 scale: Optional[float] = None,
                                 device: Optional[str] = 'cuda') -> torch.Tensor:
    """
    Computes scaled dot product attention on query, key and value tensors, using an optional attention mask if passed,
    and applying dropout if a probability greater than 0.0 is specified.
    The optional scale argument can only be specified as a keyword argument.


    Args:
        query (torch.Tensor): The query tensor of shape [batch_size, num_heads, sequence_length, d_model//num_heads].
        key (torch.Tensor): The key tensor of shape [batch_size, num_heads, sequence_length, d_model//num_heads].
        value (torch.Tensor): The value tensor of shape [batch_size, num_heads, sequence_length, d_model//num_heads].
        attn_mask (torch.Tensor, optional): The attention mask tensor, used to prevent attention from being computed on
            unwanted locations. Can be a boolean tensor or a float tensor. Default is None.
        dropout_p (float, optional): Dropout probability; if greater than 0.0, dropout is applied
        is_causal (bool, optional): If True, the attention mask is set to be a causal mask, where the attention weights
            are masked to prevent attending to future positions. Default is False.
        scale (float, optional): The scaling factor for the attention weights. If None, the scaling factor is set to
            1/sqrt(d_model//num_heads). Default is None.
        device (str, optional): The device to move the tensors to. Can be 'cpu' or 'cuda'. Default is 'cuda'

    Returns:
        torch.Tensor: The output tensor of shape [batch_size, num_heads, sequence_length, d_model//num_heads]
    """

    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype).to(device)

    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0).to(device)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not().to(device), float("-inf"))
        else:
            attn_bias += attn_mask.to(device)
    attn_weight = query @ key.transpose(-2, -1).to(device) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight.to(device), dropout_p, train=True)
    return attn_weight @ value
