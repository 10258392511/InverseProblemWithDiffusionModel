# search for ### to 1D for places to change
import torch
import torch.nn as nn


def get_normalization(config, conditional=True):
    norm = config.model.normalization
    if conditional:
        if norm == 'NoneNorm':
            return ConditionalNoneNorm1d
        elif norm == 'InstanceNorm++':
            return ConditionalInstanceNorm1dPlus
        elif norm == 'InstanceNorm':
            return ConditionalInstanceNorm1d
        elif norm == 'BatchNorm':
            return ConditionalBatchNorm1d
        elif norm == 'VarianceNorm':
            return ConditionalVarianceNorm1d
        else:
            raise NotImplementedError("{} does not exist!".format(norm))
    else:
        if norm == 'BatchNorm':
            return nn.BatchNorm1d
        elif norm == 'InstanceNorm':
            return nn.InstanceNorm1d
        elif norm == 'InstanceNorm++':
            return InstanceNorm1dPlus
        elif norm == 'VarianceNorm':
            return VarianceNorm1d
        elif norm == 'NoneNorm':
            return NoneNorm1d
        elif norm is None:
            return None
        else:
            raise NotImplementedError("{} does not exist!".format(norm))

class ConditionalBatchNorm1d(nn.Module):
    def __init__(self, num_features, num_classes, bias=True):
        super().__init__()
        self.num_features = num_features
        self.bias = bias
        self.bn = nn.BatchNorm1d(num_features, affine=False)
        if self.bias:
            self.embed = nn.Embedding(num_classes, num_features * 2)
            self.embed.weight.data[:, :num_features].uniform_()  # Initialise scale at N(1, 0.02)
            self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0
        else:
            self.embed = nn.Embedding(num_classes, num_features)
            self.embed.weight.data.uniform_()

    def forward(self, x, y):
        out = self.bn(x)
        if self.bias:
            gamma, beta = self.embed(y).chunk(2, dim=1)
            out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
        else:
            gamma = self.embed(y)
            out = gamma.view(-1, self.num_features, 1, 1) * out
        return out


class ConditionalInstanceNorm1d(nn.Module):
    def __init__(self, num_features, num_classes, bias=True):
        super().__init__()
        self.num_features = num_features
        self.bias = bias
        self.instance_norm = nn.InstanceNorm1d(num_features, affine=False, track_running_stats=False)
        if bias:
            self.embed = nn.Embedding(num_classes, num_features * 2)
            self.embed.weight.data[:, :num_features].uniform_()  # Initialise scale at N(1, 0.02)
            self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0
        else:
            self.embed = nn.Embedding(num_classes, num_features)
            self.embed.weight.data.uniform_()

    def forward(self, x, y):
        h = self.instance_norm(x)
        if self.bias:
            gamma, beta = self.embed(y).chunk(2, dim=-1)
            out = gamma.view(-1, self.num_features, 1, 1) * h + beta.view(-1, self.num_features, 1, 1)
        else:
            gamma = self.embed(y)
            out = gamma.view(-1, self.num_features, 1, 1) * h
        return out


class ConditionalVarianceNorm1d(nn.Module):
    def __init__(self, num_features, num_classes, bias=False):
        super().__init__()
        self.num_features = num_features
        self.bias = bias
        self.embed = nn.Embedding(num_classes, num_features)
        self.embed.weight.data.normal_(1, 0.02)

    def forward(self, x, y):
        vars = torch.var(x, dim=(2, 3), keepdim=True)
        h = x / torch.sqrt(vars + 1e-5)

        gamma = self.embed(y)
        out = gamma.view(-1, self.num_features, 1, 1) * h
        return out


class VarianceNorm1d(nn.Module):
    def __init__(self, num_features, bias=False):
        super().__init__()
        self.num_features = num_features
        self.bias = bias
        self.alpha = nn.Parameter(torch.zeros(num_features))
        self.alpha.data.normal_(1, 0.02)

    def forward(self, x):
        vars = torch.var(x, dim=(2, 3), keepdim=True)
        h = x / torch.sqrt(vars + 1e-5)

        out = self.alpha.view(-1, self.num_features, 1, 1) * h
        return out


class ConditionalNoneNorm1d(nn.Module):
    def __init__(self, num_features, num_classes, bias=True):
        super().__init__()
        self.num_features = num_features
        self.bias = bias
        if bias:
            self.embed = nn.Embedding(num_classes, num_features * 2)
            self.embed.weight.data[:, :num_features].uniform_()  # Initialise scale at N(1, 0.02)
            self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0
        else:
            self.embed = nn.Embedding(num_classes, num_features)
            self.embed.weight.data.uniform_()

    def forward(self, x, y):
        if self.bias:
            gamma, beta = self.embed(y).chunk(2, dim=-1)
            out = gamma.view(-1, self.num_features, 1, 1) * x + beta.view(-1, self.num_features, 1, 1)
        else:
            gamma = self.embed(y)
            out = gamma.view(-1, self.num_features, 1, 1) * x
        return out


class NoneNorm1d(nn.Module):
    def __init__(self, num_features, bias=True):
        super().__init__()

    def forward(self, x):
        return x


class InstanceNorm1dPlus(nn.Module):
    def __init__(self, num_features, bias=True):
        super().__init__()
        self.num_features = num_features
        self.bias = bias
        self.instance_norm = nn.InstanceNorm1d(num_features, affine=False, track_running_stats=False)
        self.alpha = nn.Parameter(torch.zeros(num_features))
        self.gamma = nn.Parameter(torch.zeros(num_features))
        self.alpha.data.normal_(1, 0.02)
        self.gamma.data.normal_(1, 0.02)
        if bias:
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        dim = [2 + i for i in range(len(x.shape) - 2)]
        # means = torch.mean(x, dim=(2, 3))
        means = torch.mean(x, dim=dim)
        m = torch.mean(means, dim=-1, keepdim=True)
        v = torch.var(means, dim=-1, keepdim=True)
        means = (means - m) / (torch.sqrt(v + 1e-5))
        h = self.instance_norm(x)

        if self.bias:
            ### to 1D
            # h = h + means[..., None, None] * self.alpha[..., None, None]
            # out = self.gamma.view(-1, self.num_features, 1, 1) * h + self.beta.view(-1, self.num_features, 1, 1)

            h = h + means[..., None] * self.alpha[..., None]
            out = self.gamma.view(-1, self.num_features, 1) * h + self.beta.view(-1, self.num_features, 1)
        else:
            ### to 1D
            # h = h + means[..., None, None] * self.alpha[..., None, None]
            # out = self.gamma.view(-1, self.num_features, 1, 1) * h
            h = h + means[..., None] * self.alpha[..., None]
            out = self.gamma.view(-1, self.num_features, 1) * h
        return out


class ConditionalInstanceNorm1dPlus(nn.Module):
    def __init__(self, num_features, num_classes, bias=True):
        super().__init__()
        self.num_features = num_features
        self.bias = bias
        self.instance_norm = nn.InstanceNorm1d(num_features, affine=False, track_running_stats=False)
        if bias:
            self.embed = nn.Embedding(num_classes, num_features * 3)
            self.embed.weight.data[:, :2 * num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
            self.embed.weight.data[:, 2 * num_features:].zero_()  # Initialise bias at 0
        else:
            self.embed = nn.Embedding(num_classes, 2 * num_features)
            self.embed.weight.data.normal_(1, 0.02)

    def forward(self, x, y):
        means = torch.mean(x, dim=(2, 3))
        m = torch.mean(means, dim=-1, keepdim=True)
        v = torch.var(means, dim=-1, keepdim=True)
        means = (means - m) / (torch.sqrt(v + 1e-5))
        h = self.instance_norm(x)

        if self.bias:
            ### to 1D
            gamma, alpha, beta = self.embed(y).chunk(3, dim=-1)
            # h = h + means[..., None, None] * alpha[..., None, None]
            # out = gamma.view(-1, self.num_features, 1, 1) * h + beta.view(-1, self.num_features, 1, 1)
            h = h + means[..., None] * alpha[..., None]
            out = gamma.view(-1, self.num_features, 1) * h + beta.view(-1, self.num_features, 1)
        else:
            ### to 1D
            gamma, alpha = self.embed(y).chunk(2, dim=-1)
            # h = h + means[..., None, None] * alpha[..., None, None]
            # out = gamma.view(-1, self.num_features, 1, 1) * h
            h = h + means[..., None] * alpha[..., None]
            out = gamma.view(-1, self.num_features, 1) * h
        return out
