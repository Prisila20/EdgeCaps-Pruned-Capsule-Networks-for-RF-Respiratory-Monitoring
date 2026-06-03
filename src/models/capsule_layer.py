import torch
import torch.nn as nn
import torch.nn.functional as F

class CapsuleLayer(nn.Module):
    """
    A PyTorch implementation of the CapsuleLayer analogous to the Keras/TensorFlow version.
    Expects input of shape [batch_size, input_num_capsules, input_dim_capsules]
    and produces output of shape [batch_size, num_capsules, dim_capsules].
    """
    def __init__(self, 
                 input_num_capsules: int,
                 input_dim_capsules: int,
                 num_capsules: int,
                 dim_capsules: int,
                 routings: int = 3):
        """
        Args:
            input_num_capsules: Number of capsules in the input layer.
            input_dim_capsules: Dimensionality of each input capsule.
            num_capsules: Number of capsules in this layer.
            dim_capsules: Dimensionality of each capsule in this layer.
            routings: Number of routing iterations.
        """
        super(CapsuleLayer, self).__init__()
        self.input_num_capsules = input_num_capsules
        self.input_dim_capsules = input_dim_capsules
        self.num_capsules = num_capsules
        self.dim_capsules = dim_capsules
        self.routings = routings
        
        # In TF code: W shape = [1, input_num_capsules, num_capsules, dim_capsules, input_shape[-1]]
        # We'll use shape: [1, input_num_capsules, num_capsules, dim_capsules, input_dim_capsules].
        self.W = nn.Parameter(
            torch.empty(
                1, 
                self.input_num_capsules, 
                self.num_capsules, 
                self.dim_capsules, 
                self.input_dim_capsules
            )
        )
        
        # Initialize weights (similar to Glorot/Xavier uniform)
        nn.init.xavier_uniform_(self.W)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the CapsuleLayer.

        Args:
            inputs: Tensor of shape [batch_size, input_num_capsules, input_dim_capsules]

        Returns:
            Tensor of shape [batch_size, num_capsules, dim_capsules]
        """
        batch_size = inputs.shape[0]
        # inputs => [batch_size, input_num_capsules, input_dim_capsules]

        # Expand dims to match W for batched matmul:
        # [batch_size, input_num_capsules, input_dim_capsules] 
        # -> [batch_size, input_num_capsules, 1, input_dim_capsules, 1]
        inputs_expand = inputs.unsqueeze(-1).unsqueeze(2)
        # Broadcast inputs across num_capsules dimension:
        inputs_expand = inputs_expand.expand(
            batch_size,
            self.input_num_capsules,
            self.num_capsules,
            self.input_dim_capsules,
            1
        )
        
        # Broadcast W across batch dimension:
        W_tiled = self.W.expand(
            batch_size,
            self.input_num_capsules,
            self.num_capsules,
            self.dim_capsules,
            self.input_dim_capsules
        )

        # Matrix multiplication:
        # (dim_capsules, input_dim_capsules) x (input_dim_capsules, 1) 
        # => [dim_capsules, 1]
        u_hat = torch.matmul(W_tiled, inputs_expand)
        # Now remove trailing dimension => shape = [batch_size, input_num_capsules, num_capsules, dim_capsules]
        u_hat = u_hat.squeeze(-1)

        # Initialize routing logits (b) to zero
        b = torch.zeros(
            (batch_size, self.input_num_capsules, self.num_capsules, 1),
            device=u_hat.device,
            dtype=u_hat.dtype
        )

        # Dynamic routing
        for i in range(self.routings):
            # c = softmax(b) over num_capsules dimension
            c = F.softmax(b, dim=2)

            # Weighted sum over input_num_capsules dimension
            s = (c * u_hat).sum(dim=1, keepdim=True)
            # Squash
            v = self.squash(s, dim=-1)

            if i < self.routings - 1:
                # Update b
                # Expand v to compare with each u_hat
                v_tiled = v.expand(-1, self.input_num_capsules, -1, -1)
                b = b + (u_hat * v_tiled).sum(dim=-1, keepdim=True)

        # Remove dimension for input_num_capsules => [batch_size, num_capsules, dim_capsules]
        return v.squeeze(1)

    @staticmethod
    def squash(s: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """
        Squash activation used in Capsule Networks.
        Args:
            s: Input tensor to squash.
            dim: Dimension along which to compute the norm and squash.
        Returns:
            Squashed tensor of the same shape as s.
        """
        s_norm_sq = (s ** 2).sum(dim=dim, keepdim=True)
        scale = s_norm_sq / (1.0 + s_norm_sq)
        s_norm = torch.sqrt(s_norm_sq + 1e-7)
        return scale * (s / s_norm)


if __name__ == "__main__":
    # Quick test
    batch_size = 32
    input_num_capsules = 8
    input_dim_capsules = 16
    num_capsules = 10
    dim_capsules = 16
    routings = 3

    x = torch.randn(batch_size, input_num_capsules, input_dim_capsules)
    caps_layer = CapsuleLayer(input_num_capsules, input_dim_capsules, num_capsules, dim_capsules, routings)
    output = caps_layer(x)
    print("CapsuleLayer output shape:", output.shape)  # [32, 10, 16]
