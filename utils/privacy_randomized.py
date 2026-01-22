import torch
import torch.nn as nn
import numpy as np

class RandomizedLabelPrivacy:
    def __init__(
        self,
        epsilon: float, # parameter of epsilon label dp
        mechanism: str, # = "staircase"
        sensitivity: float = 1,
        gamma: float = None,
        
        
        device: str = "cpu",
    ):
        r"""
        A privacy engine for randomizing labels.

        Arguments
            mechanism: type of the mechansim, for now normal, staircase or laplacian
        """
        self.epsilon = epsilon
        
        if gamma is None and epsilon is not None:
            gamma = 1 / (1 + np.exp(epsilon / 2))
        self.gamma = gamma
        self.device = device
        
        assert mechanism.lower() in ("gaussian", "laplace", "staircase")
        self.mechanism = mechanism
        self.sensitivity = sensitivity
        self._rng = np.random.default_rng()
    
        self.train()

    def train(self):
        self._train = True

    def eval(self):
        self._train = False

    def reset_randomizer(self):
        if self.randomizer is not None:
            self.randomizer.manual_seed(self.seed)

    def noise(self, shape):
        if not self._train or self._rng is None:
            return None
        if self.mechanism.lower() == "gaussian": 
            delta = 1e-4
            sigma = np.sqrt(2 * np.log(1.25 / delta)) * self.sensitivity / self.epsilon
            noises = self._rng.normal(loc= 0.0, scale= sigma, size=shape)
            
        elif self.mechanism.lower() == "staircase":
            sign = np.random.binomial(1, 0.5, shape)
            sign = 2 * sign - 1
            geometric_rv = self._rng.geometric(1 - np.exp(- self.epsilon), shape) - 1
            unif_rv = self._rng.random(shape)
            binary_rv = np.random.binomial(1, self.gamma / (self.gamma + (1 - self.gamma) * np.exp(- self.epsilon)), shape)
            binary_rv = -2 * binary_rv + 1
            
            noises = sign * ((1 - binary_rv) * ((geometric_rv + self.gamma * unif_rv) * self.sensitivity) +
                                   binary_rv * ((geometric_rv + self.gamma + (1 - self.gamma) * unif_rv) * self.sensitivity))
        elif self.mechanism.lower() == "laplace":  # is Laplace
            noises = self._rng.laplace(loc= 0.0, scale= self.sensitivity / self.epsilon, size= shape)

        noises = torch.from_numpy(noises).to(self.device).float()
        return noises