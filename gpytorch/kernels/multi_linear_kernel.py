#!/usr/bin/env python3

from copy import deepcopy
from typing import List, Optional, Union

import torch

from ..priors import Prior
from ..utils.generic import length_safe_zip
from .kernel import Kernel
from ..constraints.constraints import Positive
from linear_operator.operators import KroneckerProductLinearOperator
from linear_operator import to_linear_operator



class MLinKernel(Kernel):
    """
    The returned object is of type :obj:`~linear_operator.operators.KroneckerProductLinearOperator`.
    """

    def __init__(self, tensor_size, U_constraint=None, **kwargs):
        super(MLinKernel, self).__init__(**kwargs)
        self.tensor_size = tensor_size
        # ----- Initialize U1, U2, U3 ----- #
        U1 = torch.randn(tensor_size[0],tensor_size[0])
        U2 = torch.randn(tensor_size[1],tensor_size[1])
        if len(tensor_size) == 3:
            U3 = torch.randn(tensor_size[2],tensor_size[2])
        else:
            U3 = torch.randn(1,1)
        # const_U1, const_U2 = torch.sqrt(torch.linalg.norm(U1.T @ U1)), torch.sqrt(torch.linalg.norm(U2.T @ U2))
        # const_U3 = torch.sqrt(torch.linalg.norm(U3.T @ U3))
        self.register_parameter(name="raw_U1", parameter=torch.nn.Parameter(U1))
        self.register_parameter(name="raw_U2", parameter=torch.nn.Parameter(U2))
        self.register_parameter(name="raw_U3", parameter=torch.nn.Parameter(U3))

        # set the weight constraint
        if U_constraint is None:
            U_constraint = Positive()
        # register the constraint
        self.register_constraint("raw_U1", U_constraint)
        self.register_constraint("raw_U2", U_constraint)
        self.register_constraint("raw_U3", U_constraint)

    @property
    def U1(self):
        return self.raw_U1_constraint.transform(self.raw_U1)
    @property
    def U2(self):
        return self.raw_U2_constraint.transform(self.raw_U2)
    @property
    def U3(self):
        return self.raw_U3_constraint.transform(self.raw_U3)
    
    def forward(self, X, X2=None, diag=False, **params):

        U_321 = KroneckerProductLinearOperator(self.U3, KroneckerProductLinearOperator(self.U2, self.U1))
        U = (U_321 @ X.T).T
        K = U @ (U.T) 
        bigK = K/K.max() + (torch.eye(X.shape[0]) * (1e-4)).to(self.device)

        if diag:
            return bigK.diag()
        return to_linear_operator(bigK)

