from abc import ABCMeta, abstractclassmethod
import numpy as np

class DefaultParams():
    def __init__(self) -> None:
        self._beta1 = np.array([3, -2]).reshape(-1, 1)
        self._beta2 = np.array([2, 1]).reshape(-1, 1)
        self._n = 100
        self._omega = np.array([[0.1, -0.05], 
                               [-0.05, 0.2],])
    @property
    def beta1(self):
        return self._beta1
    @property
    def beta2(self):
        return self._beta2
    @property
    def n(self):
        return self._n
    @property
    def omega(self):
        return self._omega


class Sampler(metaclass=ABCMeta):
    def __init__(self, SUR_data:dict) -> None:
        self.data = SUR_data

    @abstractclassmethod
    def sample(self):
        pass


def generate_SUR(params:DefaultParams=DefaultParams()):
    n = params.n
    omega = params.omega
    w11, w12, w21, w22 = omega[0,0], omega[0,1], omega[1,0], omega[1,1]
    beta = np.concatenate((params.beta1, params.beta2), axis=0)

    x1 = np.random.rand(n,2) * 10 - 5
    x2 = np.random.rand(n,2) * 10 - 5
    X = np.block([
                [x1,              np.zeros((n, 2))],
                [np.ones((n, 2)), x2              ],
    ])
    Omega_combined = np.block([
                [w11*np.eye(n), w12*np.eye(n)],
                [w21*np.eye(n), w22*np.eye(n)],
    ])
    U = np.random.multivariate_normal(mean=np.zeros(2*n), cov=Omega_combined).reshape(-1,1)
    Y = X @ beta + U
    assert Y.shape == (2*n, 1)

    SUR_data = {}
    SUR_data['X'] = X
    SUR_data['x1'] = x1
    SUR_data['x2'] = x2
    SUR_data['Y'] = Y
    SUR_data['U'] = U
    
    return SUR_data