import secrets


def uniform(a: float = 0.0, b: float = 1.0) -> float:
    """Cryptographically secure uniform sample."""
    # Generate 53 random bits and scale to [0, 1)
    u = secrets.randbits(53) / (1 << 53)
    return a + (b - a) * u

def exponentialdist(lam: float = 1.0) -> float:
    """Generate an exponential sample using inverse transform sampling."""
    import math

    if not math.isfinite(lam) or lam <= 0:
        raise ValueError("lam must be finite and positive.")

    u = uniform()
    return -math.log1p(-u) / lam

def poissiondist(lam: float = 4.0) -> int:
    """Generate a Poisson sample using inverse CDF sampling."""
    import math

    if not math.isfinite(lam) or lam < 0:
        raise ValueError("lam must be finite and nonnegative.")

    if lam == 0:
        return 0

    u = uniform()
    k = 0
    p = math.exp(-lam)
    cumulative = p

    if p == 0:
        raise ValueError("lam is too large for this direct method.")

    while u > cumulative:
        k += 1
        p *= lam / k
        updated = cumulative + p

        if updated == cumulative:
            raise ArithmeticError("CDF calculation reached numerical precision.")

        cumulative = updated

    return k
