from typing import Optional

import torch
from torch import nn, Tensor
import copy


def _pos_embedding(x: Tensor, pos: Optional[Tensor]):
    return x if pos is None else x + pos


class TransformerEncoderLayer(nn.Module):
    def __init__(self, 
                 dmodel: int, 
                 nhead: int, dim_ff: int, 
                 dropout: float, 
                 activation: str = 'ReLU') -> None:
        super().__init__()

        act = getattr(nn, activation)
        self.attn = nn.MultiheadAttention(dmodel, nhead, dropout)
        self.mlp = nn.Sequential(
            nn.Linear(dmodel, dim_ff),
            act(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, dmodel),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)
        self.ln1, self.ln2 = (nn.LayerNorm(dmodel) for _ in range(2))
        self.dmodel = dmodel

    def forward(self, x0: Tensor, 
                attn_mask: Optional[Tensor] = None, 
                key_padding_mask: Optional[Tensor] = None, 
                pos_emb: Optional[Tensor] = None) -> torch.Tensor:
        q = k = _pos_embedding(x0, pos_emb)
        x = self.dropout(self.attn(q, k, value=x0, attn_mask=attn_mask, key_padding_mask=key_padding_mask)[0]) 
        x0 = self.ln1(x0 + x)       
        x0 = self.ln2(x0 + self.mlp(x))
        return x0
        

class TransformerDecoderLayer(nn.Module):
    def __init__(self, 
                 dmodel: int, 
                 nhead: int, 
                 dim_ff: int, 
                 dropout: float = 0.1, 
                 activation: str = 'ReLU') -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dmodel, nhead, dropout)
        self.mh_attn = nn.MultiheadAttention(dmodel, nhead, dropout)
        act = getattr(nn, activation)
        self.mlp = nn.Sequential(
            nn.Linear(dmodel, dim_ff),
            act(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, dmodel),
            nn.Dropout(dropout)
        )
        self.dropout1, self.dropout2 = (nn.Dropout(dropout) for _ in range(2))
        self.ln1, self.ln2, self.ln3 = (nn.LayerNorm(dmodel) for _ in range(3))
        self.dmodel = dmodel

    def forward(self, 
                tgt: Tensor, 
                memory: Tensor,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None) -> torch.Tensor:
        q = k = _pos_embedding(tgt, query_pos)
        x = self.dropout1(self.self_attn(q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0])
        tgt = self.ln1(tgt + x)
        x = self.mh_attn(query=_pos_embedding(tgt, query_pos),
                         key=_pos_embedding(memory, pos),
                         value=memory,
                         attn_mask=memory_mask,
                         key_padding_mask=memory_key_padding_mask)[0]
        tgt = self.ln2(tgt + self.dropout2(x))
        tgt = self.ln3(tgt + self.mlp(tgt))
        return tgt


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None) -> None:
        super().__init__()
        self.norm = norm
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, 
                x: Tensor, 
                attn_mask: Optional[Tensor] = None, 
                key_padding_mask: Optional[Tensor] = None, 
                pos_emb: Optional[Tensor] = None) -> torch.Tensor:
        output = x
        for layer in self.layers:
            output = layer(output, attn_mask=attn_mask, key_padding_mask=key_padding_mask, pos_emb=pos_emb)
        
        if self.norm is not None:
            output = self.norm(output)
        
        return output


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None) -> None:
        super().__init__()
        self.norm = norm
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, tgt: Tensor, memory: Tensor,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None) -> torch.Tensor:
        output = tgt
        for layer in self.layers:
            output = layer(output, memory, 
                           tgt_mask=tgt_mask, memory_mask=memory_mask, 
                           tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            
        if self.norm is not None:
            output = self.norm(output)

        return output.unsqueeze(0)


class SkipConnectTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None) -> None:
        super().__init__()
        self.norm = norm
        dmodel = encoder_layer.dmodel
        
        assert num_layers % 2 == 1, "num_layers must be odd to ensure U-Net structure"
        num_blocks = (num_layers - 1) // 2
        self.input_blocks = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_blocks)])
        self.middle_block = copy.deepcopy(encoder_layer)
        self.output_blocks = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_blocks)])
        self.linear_blocks = nn.ModuleList([nn.Linear(2 * dmodel, dmodel) for _ in range(num_blocks)])
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, 
                x: Tensor, 
                attn_mask: Optional[Tensor] = None, 
                key_padding_mask: Optional[Tensor] = None, 
                pos_emb: Optional[Tensor] = None) -> torch.Tensor:
        xs = []
        for module in self.input_blocks:
            x = module(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, pos_emb=pos_emb)
            xs.append(x)
        x = self.middle_block(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, pos_emb=pos_emb)
        for (module, linear) in zip(self.output_blocks, self.linear_blocks):
            x = torch.cat([x, xs.pop()], dim=-1)
            x = linear(x)
            x = module(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, pos_emb=pos_emb)

        if self.norm is not None:
            x = self.norm(x)

        return x

class SkipConnectTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None) -> None:
        super().__init__()
        dmodel = decoder_layer.dmodel
        self.norm = norm

        assert num_layers % 2 == 1, "num_layers must be odd to ensure U-Net structure"
        num_blocks = (num_layers - 1) // 2
        self.input_blocks = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_blocks)])
        self.middle_block = copy.deepcopy(decoder_layer)
        self.output_blocks = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_blocks)])
        self.linear_blocks = nn.ModuleList([nn.Linear(2 * dmodel, dmodel) for _ in range(num_blocks)])
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, 
                tgt: Tensor,
                memory: Tensor,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None) -> torch.Tensor: 
        x = tgt
        xs = []
        for module in self.input_blocks:
            x = module(x, memory, 
                       tgt_mask=tgt_mask, memory_mask=memory_mask,
                       tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask,
                       pos=pos, query_pos=query_pos)
            xs.append(x)

        x = self.middle_block(x, memory, 
                              tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask, 
                              pos=pos, query_pos=query_pos)

        for module, linear in zip(self.output_blocks, self.linear_blocks):
            x = torch.cat([x, xs.pop()], dim=-1)
            x = linear(x)
            x = module(x, memory, 
                       tgt_mask=tgt_mask, memory_mask=memory_mask,
                       tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask,
                       pos=pos, query_pos=query_pos)

        if self.norm is None:
            x = self.norm(x)

        return x