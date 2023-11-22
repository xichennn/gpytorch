#!/usr/bin/env python3

from typing import Optional

import torch
import torch.nn.functional as F
import numpy as np
# from ..settings import lazily_evaluate_kernels
from .kernel import Kernel
from linear_operator import to_dense, to_linear_operator
from ..constraints.constraints import Positive

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
        self.padding = 0

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
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()  # [799, 6, 4, 1, 3, 2]
        patches = patches.reshape(X.shape[0], self.num_patches, -1)
        
        return patches.float() # [800, 24, 6]
        
    def forward(self, X, X2=None, diag=False, **params):
        # if (
        #     X.requires_grad
        #     or X2.requires_grad
        #     or (self.ard_num_dims is not None and self.ard_num_dims > 1)
        #     or diag
        #     or params.get("last_dim_is_batch", False)
        #     or trace_mode.on()
        # ):
        if diag:
            return self.Kdiag(X)
       
        Xp = self._get_patches(X)
        Xp = Xp.view(-1, self.patch_len) #(N*num_patches, patch_h*patch_w)
        Xp2 = self._get_patches(X2).view(-1, self.patch_len) if X2 is not None else None
     
        bigK = self.base_kernel(Xp, Xp2)
        K = torch.mean(bigK.to_dense().view(X.size(0), self.num_patches, -1, self.num_patches), dim=(1, 3))
      
        return to_linear_operator(K)

    def Kdiag(self, X):
        Xp = self._get_patches(X)

        def sumbK(Xp):
            return torch.sum(self.base_kernel(Xp))

        return torch.stack([sumbK(patch) for patch in Xp]) / self.num_patches ** 2.0

    def Kzx(self, Z, X):
        Xp = self._get_patches(X)
        Xp = Xp.view(-1, self.patch_len)
        bigKzx = self.base_kernel.K(Z, Xp)
        Kzx = torch.sum(bigKzx.view(Z.size(0), X.size(0), self.num_patches), [2])
        return Kzx / self.num_patches

    def Kzz(self, Z):
        return self.base_kernel.K(Z)

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

class ColourPatchConv(ConvKernel):
    def __init__(self, base_kernel, img_size, patch_size, colour_channels=1, **kwargs):
        super(ColourPatchConv, self).__init__(base_kernel, img_size, patch_size, colour_channels, **kwargs)
        self.base_kernel.input_dim = np.prod(patch_size) * self.colour_channels

    def _get_patches(self, X):
        castX = X.view(X.size(0), self.img_size[0], self.img_size[1], self.colour_channels)
        # patches = F.unfold(castX, self.patch_size, padding=0)
        # get all image windows of size (patch_h, patch_w) and stride (1,1)
        patches = castX.unfold(1, self.patch_size[0], 1).unfold(2, self.patch_size[1], 1)
        # Permute so that channels are next to patch dimension
        patches = patches.contiguous()  # [799, 6, 4, 1, 3, 2]
        return patches.view(X.size(0), self.num_patches, -1)

    @property
    def patch_len(self):
        return np.prod(self.patch_size) * self.colour_channels

    @property
    def num_patches(self):
        return (self.img_size[0] - self.patch_size[0] + 1) * (self.img_size[1] - self.patch_size[1] + 1)


class WeightedConv(ConvKernel):
    def __init__(self, base_kernel, img_size, patch_size, colour_channels=1, weight_constraint=None, **kwargs):
        super(WeightedConv, self).__init__(base_kernel, img_size, patch_size, colour_channels, **kwargs)
        # self.W = GPflow.param.Param(np.ones(self.num_patches))
        self.register_parameter(name="raw_patch_weights", parameter=torch.nn.Parameter(torch.ones(self.num_patches)))
        # set the weight constraint
        if weight_constraint is None:
            weight_constraint = Positive()
        # register the constraint
        self.register_constraint("raw_patch_weights", weight_constraint)

    # now set up the 'actual' paramter
    @property
    def patch_weights(self):
        # when accessing the parameter, apply the constraint transform
        return self.raw_patch_weights_constraint.transform(self.raw_patch_weights)
    
    def forward(self, X, X2=None, diag=False, **params):
        if diag:
            return self.Kdiag(X)
        Xp = self._get_patches(X)
        Xp = Xp.view(-1, self.patch_len)
        Xp2 = None if X2 is None else self._get_patches(X2).view(X2.size(0) * self.num_patches, self.patch_len)

        bigK = self.base_kernel(Xp, Xp2)
        bigK = bigK.to_dense().view(X.size(0), self.num_patches, -1, self.num_patches)
        W2 = self.patch_weights.unsqueeze(0) * self.patch_weights.unsqueeze(1)
        W2 = W2.unsqueeze(0)
        W2 = W2.unsqueeze(2)
        W2bigK = bigK * W2
        K = torch.sum(W2bigK, [1, 3]) / self.num_patches ** 2.0
        return to_linear_operator(K)

    def Kdiag(self, X):
        Xp = self._get_patches(X)
        W2 = self.patch_weights.unsqueeze(0) * self.patch_weights.unsqueeze(1)

        def Kdiag_element(patches):
            return torch.sum(self.base_kernel(patches) * W2)

        return torch.stack([Kdiag_element(patch) for patch in Xp]) / self.num_patches ** 2.0

    def Kzx(self, Z, X):
        Xp = self._get_patches(X)
        Xp = Xp.view(-1, self.patch_len)
        bigKzx = self.base_kernel(Z, Xp)
        bigKzx = bigKzx.view(Z.size(0), X.size(0), self.num_patches)
        Kzx = torch.sum(bigKzx * self.W, [2])
        return Kzx / self.num_patches
