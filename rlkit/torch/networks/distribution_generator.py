import abc

from torch import nn

from rlkit.util.distributions import Distribution
# TODO: clear these imports
# MultivariateDiagonalNormal,
# from rlkit.torch.networks.basic import MultiInputSequential


class DistributionGenerator(nn.Module, metaclass=abc.ABCMeta):
    def forward(self, *input, **kwarg) -> Distribution:
        raise NotImplementedError


# TODO: clear this out if we don't need it.
# class ModuleToDistributionGenerator(
#     MultiInputSequential,
#     DistributionGenerator,
#     metaclass=abc.ABCMeta
# ):
#     pass


# class Gaussian(ModuleToDistributionGenerator):
#     def __init__(self, module, std=None, reinterpreted_batch_ndims=1):
#         super().__init__(module)
#         self.std = std
#         self.reinterpreted_batch_ndims = reinterpreted_batch_ndims

#     def forward(self, *input):
#         if self.std:
#             mean = super().forward(*input)
#             std = self.std
#         else:
#             mean, log_std = super().forward(*input)
#             std = log_std.exp()
#         return MultivariateDiagonalNormal(
#             mean, std,
#             reinterpreted_batch_ndims=self.reinterpreted_batch_ndims)
