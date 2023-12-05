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
        a = torch.arange(0, v).view(-1, 1).repeat(h, 1).view(-1, 1).repeat(w, 1)
        b = torch.arange(0, h).view(-1, 1).repeat(1, v).view(-1, 1).repeat(w, 1)
        c = torch.arange(0, w).view(-1, 1).repeat(1, v*h).view(-1, 1)
        # Combining tensors a, b, c to get Vloc
        Vloc = torch.cat((a, b, c), dim=1).to(self.device)
        Vloc = Vloc.to(torch.float32)
        Vloc_ = Vloc.div(self.IMED_lengthscale)
        
        # pixel pairwise Euclidean distance on image lattice
        Z = self.covar_dist(Vloc_, Vloc_, square_dist=True)  #try torch.cdist(Vloc, Vloc)**2
        G = 1/(2*math.pi*self.IMED_lengthscale**2)*torch.exp(-Z/2) 
        G = G + 1e-6*torch.diag(torch.ones(G.shape[0])).to(self.device)

        return G

    def forward(self, X, X2=None, diag=False, **params):
        if diag:
            return self.Kdiag(X, diag)
        G = self._get_G()
        #G=L@L.mT
        L = torch.linalg.cholesky(G)
        Xp = X@L
        Xp2 = X2@L if X2 is not None else None

        return self.base_kernel(Xp, Xp2)
        
    def Kdiag(self, X, diag):
        G = self._get_G()
        L = torch.linalg.cholesky(G)
        Xp = X@L

        return self.base_kernel(Xp, diag=diag)
        






