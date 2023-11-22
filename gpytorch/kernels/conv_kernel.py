#!/usr/bin/env python3

from typing import Optional

import torch
import torch.nn.functional as F
import numpy as np
from .kernel import Kernel
from linear_operator import to_dense, to_linear_operator
from ..constraints.constraints import Positive

import time

class ConvKernel(Kernel):
    def __init__(self, base_kernel, img_size, patch_size, colour_channels=1, **kwargs):
        super(ConvKernel, self).__init__(**kwargs)
        self.img_size = img_size
        self.patch_size = patch_size
        self.base_kernel = base_kernel
        self.base_kernel.input_dim = np.prod(patch_size)
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

    @property
    def patch_len(self):
        return np.prod(self.patch_size)

    @property
    def num_patches(self):
        # return (self.img_size[0] - self.patch_size[0] + 1) * (
        #     self.img_size[1] - self.patch_size[1] + 1) * self.colour_channels
        return (self.img_size[0]+2*self.padding-self.patch_size[0] + 1) * (
               self.img_size[1]+2*self.padding-self.patch_size[1] + 1) * self.colour_channels

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

class GeometricKernel(Kernel):
    def __init__(self, base_kernel, tensor_size, voxel_size, num_properties=1, padding=0, **kwargs):
        # super(ConvKernel, self).__init__(np.prod(img_size))
        super(GeometricKernel, self).__init__(**kwargs)
        self.tensor_size = tensor_size #[V,H,W]
        self.voxel_size = voxel_size #[v,h,w]
        self.base_kernel = base_kernel
        self.base_kernel.input_dim = np.prod(voxel_size)
        # self.base_kernel.ard_num_dims = np.prod(patch_size)
        self.num_properties = num_properties
        self.padding = padding

    def _get_voxels(self, X):
        """
        Extracts patches from the images X. Patches are extracted separately for each of the colour channels.
        :param X: (N x input_dim)
        :return: Voxels (N, num_voxels, voxel_size)
        """
        castX = X.view(-1, self.tensor_size[0], self.tensor_size[1], self.tensor_size[2], self.num_properties) #only 4D tensors [N, C, *] are supported
        # Pad tensor to get the same output
        # out_size ​= ⌊(input_size​+2×padding−dilation×(kernel_size−1)−1)/stride ​+ 1⌋

        # get all voxels of size (v,h,w) and stride (1,1)
        voxels = castX.unfold(1, self.voxel_size[0], 1).unfold(2, self.voxel_size[1], 1).unfold(3, self.voxel_size[2], 1)
        voxels = voxels.contiguous()  
        # View as [batch_size, channels*patch_h*patch_w, heihgt*width]
        voxels = voxels.reshape(X.shape[0], self.num_voxels, -1)
        
        return voxels.float() # [800, 32, 18]
    
    def forward(self, X, X2=None, diag=False, **params):
        if diag:
            return self.Kdiag(X)
        Xp = self._get_voxels(X)
        Xp = Xp.view(-1, self.voxel_len) #(N*num_patches, patch_h*patch_w)
        Xp2 = self._get_voxels(X2).view(-1, self.voxel_len) if X2 is not None else None
        
        bigK = self.base_kernel(Xp, Xp2)
        K = torch.mean(bigK.to_dense().view(X.size(0), self.num_voxels, -1, self.num_voxels), dim=(1, 3))

        return to_linear_operator(K)

    def Kdiag(self, X):
        Xp = self._get_voxels(X)

        def sumbK(Xp):
            return torch.sum(self.base_kernel(Xp))

        return torch.stack([sumbK(voxel) for voxel in Xp]) / self.num_voxels ** 2.0
    @property
    def voxel_len(self):
        return np.prod(self.voxel_size)

    @property
    def num_voxels(self):
        # return (self.img_size[0] - self.patch_size[0] + 1) * (
        #     self.img_size[1] - self.patch_size[1] + 1) * self.colour_channels
        return (self.tensor_size[0]+2*self.padding-self.voxel_size[0] + 1) * (
               self.tensor_size[1]+2*self.padding-self.voxel_size[1] + 1) * (
               self.tensor_size[2]+2*self.padding-self.voxel_size[2] + 1) * self.num_properties
    
class WeightedGeometric(GeometricKernel):
    def __init__(self, base_kernel, tensor_size, voxel_size, num_properties=1, weight_constraint=None, **kwargs):
        super(WeightedGeometric, self).__init__(base_kernel, tensor_size, voxel_size, num_properties, **kwargs)
        # self.W = GPflow.param.Param(np.ones(self.num_patches))
        self.register_parameter(name="raw_voxel_weights", parameter=torch.nn.Parameter(torch.ones(self.num_voxels)))
        # set the weight constraint
        if weight_constraint is None:
            weight_constraint = Positive()
        # register the constraint
        self.register_constraint("raw_voxel_weights", weight_constraint)

    # now set up the 'actual' paramter
    @property
    def voxel_weights(self):
        # when accessing the parameter, apply the constraint transform
        return self.raw_voxel_weights_constraint.transform(self.raw_voxel_weights)
    
    def forward(self, X, X2=None, diag=False, **params):
        if diag:
            return self.Kdiag(X)
        Xp = self._get_voxels(X)
        Xp = Xp.view(-1, self.voxel_len)
        Xp2 = None if X2 is None else self._get_voxels(X2).view(X2.size(0) * self.num_voxels, self.voxel_len)

        bigK = self.base_kernel(Xp, Xp2)
        bigK = bigK.to_dense().view(X.size(0), self.num_voxels, -1, self.num_voxels)
        W2 = self.voxel_weights.unsqueeze(0) * self.voxel_weights.unsqueeze(1)
        W2 = W2.unsqueeze(0)
        W2 = W2.unsqueeze(2)
        W2bigK = bigK * W2
        K = torch.sum(W2bigK, [1, 3]) / self.num_voxels ** 2.0
        return to_linear_operator(K)

    def Kdiag(self, X):
        Xp = self._get_voxels(X)
        W2 = self.voxel_weights.unsqueeze(0) * self.voxel_weights.unsqueeze(1)

        def Kdiag_element(voxels):
            return torch.sum(self.base_kernel(voxels) * W2)

        return torch.stack([Kdiag_element(voxel) for voxel in Xp]) / self.num_voxels ** 2.0
