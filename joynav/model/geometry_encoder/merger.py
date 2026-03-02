import torch
import torch.nn as nn

class GeometryPatchMerger(nn.Module):
    """Unified merger for geometry features from different encoders.
    
    Supports different merger types:
    - "mlp": MLP-based feature transformation with spatial merging
    - "avg": Average pooling across spatial merge dimensions
    - "attention": Attention-based merger (not implemented yet)
    """
    
    def __init__(
        self, 
        in_hidden_size: int,
        hidden_size: int, 
        out_hidden_size: int, 
        spatial_merge_size: int = 2, 
        merger_type: str = "mlp",
        use_postshuffle_norm: bool = True
    ):
        super().__init__()
        self.in_hidden_size = in_hidden_size * (spatial_merge_size**2)
        self.mid_hidden_size = hidden_size
        self.out_hidden_size = out_hidden_size
        self.merge_size = spatial_merge_size
        self.merger_type = merger_type
        self.use_postshuffle_norm = use_postshuffle_norm
        
        if merger_type == "mlp":
            self.norm = nn.LayerNorm(self.in_hidden_size if use_postshuffle_norm else in_hidden_size, eps=1e-6)
            self.linear_fc1 = nn.Linear(self.in_hidden_size, self.mid_hidden_size)
            self.act_fn = nn.GELU()
            self.linear_fc2 = nn.Linear(self.mid_hidden_size, out_hidden_size)
            nn.init.zeros_(self.linear_fc2.weight)
            nn.init.zeros_(self.linear_fc2.bias)
        else:
            raise ValueError(f"Unknown merger type: {merger_type}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the merger."""

        n_image, h_patch, w_patch, dim = x.shape
        assert h_patch % self.merge_size == 0 and w_patch % self.merge_size == 0, \
            f"Height and width of patches ({h_patch}, {w_patch}) must be divisible by merge size {self.merge_size}"
        # x = x[:, :h_patch // self.merge_size * self.merge_size, :w_patch // self.merge_size*self.merge_size , :]
        x = x.reshape(n_image, h_patch // self.merge_size, self.merge_size, w_patch // self.merge_size, self.merge_size, dim)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()

        if self.merger_type == "mlp":
            x = self.norm(x.view(-1, self.in_hidden_size) if self.use_postshuffle_norm else x).view(-1, self.in_hidden_size)
            x = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
        else:
            raise NotImplementedError(f"Merger type {self.merger_type} not implemented")
        x = x.reshape(n_image, h_patch // self.merge_size, w_patch // self.merge_size, -1)
        return x
