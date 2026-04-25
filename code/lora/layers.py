import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, rank: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.merged = False
        
        for param in self.base_layer.parameters():
            param.requires_grad = False
        
        in_features = base_layer.in_features
        out_features = base_layer.out_features
        
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = self.base_layer(x)
        if self.merged:
            return base_output
        x_dropped = self.lora_dropout(x)
        lora_output = (x_dropped @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base_output + lora_output
    
    @torch.no_grad()
    def merge(self):
        """Fold LoRA weights into base layer for zero-latency inference."""
        if self.merged:
            return
        delta = (self.lora_B @ self.lora_A) * self.scaling
        self.base_layer.weight.data += delta
        self.merged = True
    
    @torch.no_grad()
    def unmerge(self):
        if not self.merged:
            return
        delta = (self.lora_B @ self.lora_A) * self.scaling
        self.base_layer.weight.data -= delta
        self.merged = False

class LoRAConv1D(nn.Module):
    """LoRA for HuggingFace GPT-2's Conv1D (weight shape: [in, out], transposed vs nn.Linear)."""
    
    def __init__(self, base_layer, rank: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.scaling = alpha / rank
        self.merged = False
        
        for param in self.base_layer.parameters():
            param.requires_grad = False
        
        # Conv1D stores weight as [in_features, out_features]
        in_features, out_features = base_layer.weight.shape
        
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x):
        base_output = self.base_layer(x)
        if self.merged:
            return base_output
        x_dropped = self.lora_dropout(x)
        lora_output = (x_dropped @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base_output + lora_output

    @torch.no_grad()
    def merge(self):
        if self.merged:
            return
        # Conv1D weight has shape [in, out]; LoRA delta = (B @ A) * scaling has shape [out, in].
        delta = (self.lora_B @ self.lora_A) * self.scaling
        self.base_layer.weight.data += delta.T
        self.merged = True

    @torch.no_grad()
    def unmerge(self):
        if not self.merged:
            return
        delta = (self.lora_B @ self.lora_A) * self.scaling
        self.base_layer.weight.data -= delta.T
        self.merged = False


class LoRAConv1DQV(nn.Module):
    """LoRA on Q and V slices of GPT-2's fused c_attn Conv1D.

    c_attn outputs [Q | K | V] each of size hidden. The LoRA paper applies
    adapters to W_q and W_v only — this layer holds two independent rank-r
    pairs and leaves the K slice untouched.
    """

    def __init__(self, base_layer, rank: int = 4, alpha: int = 32, dropout: float = 0.0):
        super().__init__()
        self.base_layer = base_layer
        for param in self.base_layer.parameters():
            param.requires_grad = False

        in_features, out_features = base_layer.weight.shape  # [hidden, 3*hidden]
        if out_features % 3 != 0:
            raise ValueError(f"c_attn out_features={out_features} not divisible by 3")
        hidden = out_features // 3

        self.in_features = in_features
        self.hidden = hidden
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.merged = False

        self.lora_A_q = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B_q = nn.Parameter(torch.zeros(hidden, rank))
        self.lora_A_v = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B_v = nn.Parameter(torch.zeros(hidden, rank))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A_q, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A_v, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B_q)
        nn.init.zeros_(self.lora_B_v)

    def forward(self, x):
        base_output = self.base_layer(x)
        if self.merged:
            return base_output
        x_dropped = self.lora_dropout(x)
        delta_q = (x_dropped @ self.lora_A_q.T @ self.lora_B_q.T) * self.scaling
        delta_v = (x_dropped @ self.lora_A_v.T @ self.lora_B_v.T) * self.scaling
        zeros_k = torch.zeros_like(delta_q)
        delta = torch.cat([delta_q, zeros_k, delta_v], dim=-1)
        return base_output + delta

    @torch.no_grad()
    def merge(self):
        if self.merged:
            return
        delta_q = (self.lora_B_q @ self.lora_A_q) * self.scaling  # [hidden, in]
        delta_v = (self.lora_B_v @ self.lora_A_v) * self.scaling
        h = self.hidden
        self.base_layer.weight.data[:, :h] += delta_q.T
        self.base_layer.weight.data[:, 2 * h:] += delta_v.T
        self.merged = True

    @torch.no_grad()
    def unmerge(self):
        if not self.merged:
            return
        delta_q = (self.lora_B_q @ self.lora_A_q) * self.scaling
        delta_v = (self.lora_B_v @ self.lora_A_v) * self.scaling
        h = self.hidden
        self.base_layer.weight.data[:, :h] -= delta_q.T
        self.base_layer.weight.data[:, 2 * h:] -= delta_v.T
        self.merged = False