"""Routing & mixing matrix with latent-driven weight morphing."""
import numpy as np

class RoutingMatrix:
    def __init__(self):
        self.weights = np.array([0.5, 0.3, 0.2])

    def blend(self, amp_out, fx_out, phys_out, length):
        def _fit(x):
            x = np.asarray(x)
            return x[:length] if len(x) >= length else np.pad(x, (0, length - len(x)))
        return (self.weights[0] * _fit(amp_out) +
                self.weights[1] * _fit(fx_out) +
                self.weights[2] * _fit(phys_out))

    def morph(self, latent_vector):
        expr = latent_vector[16:24]
        self.weights = np.array([0.3 + 0.4 * expr[0],
                                 0.1 + 0.5 * expr[1],
                                 0.1 + 0.4 * expr[2]])
        self.weights /= self.weights.sum() + 1e-9
        return self.weights
