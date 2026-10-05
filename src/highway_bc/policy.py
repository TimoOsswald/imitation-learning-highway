import torch
from torch import nn


class BCPolicy(nn.Module):
    """
    Behavior cloning policy: maps a driving situation to action scores.

    MLP with two hidden layers. Takes an observation of driving-situation and
    returns one logit per action in highway-env.

    Args:
        hidden: width of the hidden layers; 128 by default but subject to change after tryout
    """

    def __init__(self, hidden=128):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(     # Sequential passes the data through the layers in order
            # linear matrix multiplication: y = x * W_T + b
            # [B, 25] -> [B, hidden]
            nn.Linear(25, hidden),
            # non-linearity; without it all Linear layers would collapse into a single one
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            # output layer, one score per action; no Softmax here,
            # CrossEntropyLoss applies it internally
            nn.Linear(hidden, 5),
        )

    def forward(self, x):
        """
        Runs one forward pass through the network.

        Args:
            x: float tensor of shape [B, 25], batch of flattened observations

        Returns:
            logits: float tensor of shape [B, 5], raw action scores
                (apply Softmax only if you need probabilities)
        """
        logits = self.linear_relu_stack(x)
        return logits