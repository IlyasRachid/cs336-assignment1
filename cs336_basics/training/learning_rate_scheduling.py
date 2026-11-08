import math

def cosine_schedule(t: int, alpha_max: float, alpha_min: float, Tw: int, Tc: int) -> float:
    assert Tc > Tw, "Tc must be strictly bigger than Tw"
    if t < Tw:
        return t / Tw * alpha_max
    if t <= Tc:
        return alpha_min + 0.5 * (1 + math.cos((t - Tw) / (Tc - Tw) * math.pi)) * (alpha_max - alpha_min)
    return alpha_min