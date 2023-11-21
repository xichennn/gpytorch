#!/usr/bin/env python3

from typing import Optional

import torch
import torch.nn.functional as F
import numpy as np
# from ..settings import lazily_evaluate_kernels
from .kernel import Kernel
from linear_operator import to_dense, to_linear_operator
# import sys
# import os

# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
# from gpytorch.kernels.kernel import Kernel
import time

class ConvKernel(Kernel):
    def __init__(self, base_kernel, img_size, patch_size, colour_channels=1, **kwargs):
        # super(ConvKernel, self).__init__(np.prod(img_size))
        super(ConvKernel, self).__init__(**kwargs)
        self.img_size = img_size
        self.patch_size = patch_size
        self.base_kernel = base_kernel
        self.base_kernel.input_dim = np.prod(patch_size)
        # self.base_kernel.ard_num_dims = np.prod(patch_size)
        self.colour_channels = colour_channels
        self.padding = 1

    def _get_patches(self, X):
        """
        Extracts patches from the images X. Patches are extracted separately for each of the colour channels.
        :param X: (N x input_dim)
        :return: Patches (N, num_patches, patch_size)
        """
        castX = X.view(-1, self.colour_channels, self.img_size[0], self.img_size[1]) #only 4D tensors [N, C, *] are supported
        # Pad tensor to get the same output
        # out_size ​= ⌊(input_size​+2×padding−dilation×(kernel_size−1)−1)/stride ​+ 1⌋
        castX = F.pad(castX, (self.padding, self.padding, self.padding, self.padding))

        # get all image windows of size (patch_h, patch_w) and stride (1,1)
        patches = castX.unfold(2, self.patch_size[0], 1).unfold(3, self.patch_size[1], 1)
        # Permute so that channels are next to patch dimension
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()  # [80, 5, 5, 1, 3, 3]
        # View as [batch_size, channels*patch_h*patch_w, heihgt*width]
        patches = patches.reshape(X.shape[0], self.colour_channels*self.patch_size[0]*self.patch_size[1], -1)
        
        return patches.float() # [800, 6, 24]
    def forward(self, X, X2=None, **params):
        # if (
        #     X.requires_grad
        #     or X2.requires_grad
        #     or (self.ard_num_dims is not None and self.ard_num_dims > 1)
        #     or diag
        #     or params.get("last_dim_is_batch", False)
        #     or trace_mode.on()
        # ):
        tic = time.perf_counter()
        Xp = self._get_patches(X)
        Xp = Xp.view(-1, self.patch_len) #(N*num_patches, patch_h*patch_w)
        Xp2 = self._get_patches(X2).view(-1, self.patch_len) if X2 is not None else None
        
        # self.base_kernel.batch_shape = Xp.shape[0]
        # lazily_evaluate_kernels.off()
        bigK = self.base_kernel(Xp, Xp2)
        toc = time.perf_counter()
        K = torch.mean(bigK.to_dense().view(X.size(0), self.num_patches, -1, self.num_patches), dim=(1, 3))
        print(f"get bigK in {toc - tic:0.4f} seconds")
        return to_linear_operator(K)

    def Kdiag(self, X):
        Xp = self._get_patches(X)

        def sumbK(Xp):
            return torch.sum(self.base_kernel.K(Xp))

        return torch.stack([sumbK(patch) for patch in Xp]) / self.num_patches ** 2.0

    def Kzx(self, Z, X):
        Xp = self._get_patches(X)
        Xp = Xp.view(-1, self.patch_len)
        bigKzx = self.base_kernel.K(Z, Xp)
        Kzx = torch.sum(bigKzx.view(Z.size(0), X.size(0), self.num_patches), [2])
        return Kzx / self.num_patches

    def Kzz(self, Z):
        return self.base_kernel.K(Z)

    def init_inducing(self, X, M, method="default"):
        if method == "default" or method == "random":
            patches = self.compute_patches(X[np.random.permutation(len(X))[:M], :]).reshape(-1, self.patch_len)
            Zinit = patches[np.random.permutation(len(patches))[:M], :]
            Zinit += np.random.rand(*Zinit.shape) * 0.001
            return Zinit
        elif method == "patches-unique":
            patches = np.unique(self.compute_patches(
                X[np.random.permutation(len(X))[:M], :]).reshape(-1, self.patch_len), axis=0)
            return patches[np.random.permutation((len(patches)))[:M], :]
        else:
            raise NotImplementedError

    @property
    def patch_len(self):
        return np.prod(self.patch_size)

    @property
    def num_patches(self):
        # return (self.img_size[0] - self.patch_size[0] + 1) * (
        #     self.img_size[1] - self.patch_size[1] + 1) * self.colour_channels
        return (self.img_size[0]+2*self.padding-self.patch_size[0] + 1) * (
               self.img_size[1]+2*self.padding-self.patch_size[1] + 1) * self.colour_channels

    def compute_patches(self, X):
        return self._get_patches(X)

    def compute_Kzx(self, Z, X):
        return self.Kzx(Z, X)
