import torch
import torch.nn as nn


class BatchNorm2d(nn.Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, affine=True):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        if self.affine:
            self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1))
            self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.register_buffer("running_mean", torch.zeros(1, dim, 1, 1))
        self.register_buffer("running_var", torch.ones(1, dim, 1, 1))

    def forward(self, x):
        # 1., Calculate Statistic(mu, var) channel-wise
        if self.training:
            var, mu = torch.var_mean(x, dim=(0, 2, 3), unbiased=False, keepdim=True)
            self.running_mean = (
                self.momentum * self.running_mean + (1 - self.momentum) * mu
            )
            self.running_var = (
                self.momentum * self.running_var + (1 - self.momentum) * var
            )
        else:  # if inference, use running statistics calculated in training
            mu = self.running_mean
            var = self.running_var

        # 2. Normalize
        x = (x - mu) / torch.sqrt(var + self.eps)

        # 3. Scale and Shift
        if self.affine:
            x = x * self.gamma + self.beta
        return x
