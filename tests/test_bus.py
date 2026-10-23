import numpy as np
from aura_g.latent_bus import LatentBus, PITCH_GESTURE, TIMBRE

def test_bus_dimensions():
    bus = LatentBus()
    assert bus.vector.shape == (32,)
    bus.write_pitch_gesture(0.1, 1.0, 0.0, 0.5, 0.1, 0.3, 0.8, 0.0)
    assert bus.vector[PITCH_GESTURE][1] == 1.0
    bus.write_timbre(0.2, 0.4, 0.1, 0.05, 0.8, 0.3, 0.6, 0.7)
    assert bus.vector[TIMBRE][0] == 0.2
    print("test_bus_dimensions passed")

if __name__ == "__main__":
    test_bus_dimensions()
