import torch
from .kernel import Kernel
from ..constraints.constraints import Positive


class IMEDKernel(Kernel):
    def __init__(self, base_kernel, tensor_size, length_constraint=None, **kwargs):
        super(IMEDKernel, self).__init__(**kwargs)
        self.base_kernel = base_kernel
        self.tensor_size = tensor_size

        self.register_parameter(name="raw_IMED_lengthscale", parameter=torch.nn.Parameter(torch.zeros(1)))
        # set the weight constraint
        if length_constraint is None:
            length_constraint = Positive()
        # register the constraint
        self.register_constraint("raw_IMED_lengthscale",length_constraint)

    # now set up the 'actual' paramter
    @property
    def IMED_lengthscale(self):
        # when accessing the parameter, apply the constraint transform
        return self.raw_IMED_lengthscale_constraint.transform(self.raw_IMED_lengthscale)

    # get IMED metric G
    def _get_G(self):
        if len(self.tensor_size) == 2:
            w = 1
        elif len(self.tensor_size) == 3:
            w = self.tensor_size[2]
        v, h = self.tensor_size[0], self.tensor_size[1]
        a = torch.linspace(0,v-1,v)
        a = a.repeat(h*w)
        b = torch.linspace(0,h-1,h)
        b = b.repeat(v*w)
        c = torch.linspace(0,w-1,w)
        c = c.repeat(v*h)
        # all pixel locations
        Vloc = torch.cat((a.unsqueeze(1),b.unsqueeze(1), c.unsqueeze(1)), dim=1)
        # pixel pairwise Euclidean distance on image lattice
        Z = self.covar_dist(Vloc, Vloc)  #try torch.cdist(Ploc, Ploc)
        G = torch.exp(-self.IMED_lengthscale*Z**2) 
        G = G + 1e-6*torch.diag(torch.ones(G.shape[0]))

        return G

    def forward(self, X, X2=None, diag=False, **params):
        
        G = self._get_G()
        #G=L@L.mT
        L = torch.linalg.cholesky(G)
        Xp = X@L
        Xp2 = X2@L if X2 is not None else None

        return self.base_kernel(Xp, Xp2)
        
    def Kdiag(self, X):
        G = self._get_G()
        L = torch.linalg.cholesky(G)
        Xp = X@L

        return self.base_kernel(Xp)
        






