import torch
import torch.nn.functional as F
from torch import nn


class DropBlock1D(nn.Module):
    """
    Args:
        drop_prob (float): probability of an element to be dropped.
        block_size (int): size of the block to drop
    Shape:
        - Input: `(N, C, S)`
        - Output: `(N, C, S)`
    .. _DropBlock: A regularization method for convolutional networks:
       https://arxiv.org/abs/1810.12890
    """

    def __init__(self, drop_prob, block_size):
        super(DropBlock1D, self).__init__()

        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, x):
        # shape: (bsize, channels, series)

        assert (
            x.dim() == 3
        ), "Expected input with 3 dimensions (bsize, channels, series)"

        if not self.training or self.drop_prob == 0.0:
            return x
        else:
            # get gamma value
            gamma = self._compute_gamma(x)

            # sample mask
            mask = (torch.rand(x.shape[0], x.shape[2]) < gamma).float()

            # place mask on input device
            mask = mask.to(x.device)

            # compute block mask
            block_mask = self._compute_block_mask(mask)

            # apply block mask
            out = x * block_mask[:, None, :]

            # scale output
            out = out * block_mask.numel() / block_mask.sum()

            return out

    def _compute_block_mask(self, mask):
        block_mask = F.max_pool1d(
            input=mask[:, None, :],
            kernel_size=self.block_size,
            stride=1,
            padding=self.block_size // 2,
        )

        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1]

        block_mask = 1 - block_mask.squeeze(1)

        return block_mask

    def _compute_gamma(self, x):
        return self.drop_prob / (self.block_size ** 2)
