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