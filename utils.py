from abc import ABCMeta, abstractclassmethod
import numpy as np

class Sampler(metaclass=ABCMeta):
    def __init__(self, SUR_data:dict) -> None:
        self.data = SUR_data

    @abstractclassmethod
    def sample(self):
        pass


def generate_SUR(n:int=100):
    w11, w12, w22 = 0.1, -0.05, 0.2
    beta = np.array([3, -2, 2, 1]).reshape(-1,1)

    x1 = np.random.rand(n,2) * 10 - 5
    x2 = np.random.rand(n,2) * 10 - 5
    X = np.block([
                [x1,              np.zeros((n, 2))],
                [np.ones((n, 2)), x2              ],
    ])
    Sigma_combined = np.block([
                [w11*np.eye(n), w12*np.eye(n)],
                [w12*np.eye(n), w22*np.eye(n)],
    ])
    U = np.random.multivariate_normal(mean=np.zeros(2*n), cov=Sigma_combined).reshape(-1,1)
    Y = X @ beta + U
    assert Y.shape == (2*n, 1)

    SUR_data = {}
    SUR_data['X'] = X
    SUR_data['Y'] = Y
    SUR_data['U'] = U
    
    return SUR_data