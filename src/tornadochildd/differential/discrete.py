def diff(t, x):
    # Checking  equality of length
    if len(t) != len(x):
        raise ValueError("t and x must have the same length")
    v = []
    # Now we will compute  discrete derivative
    for k in range(1, len(t)):
        derivative = (x[k] - x[k - 1]) / (t[k] - t[k - 1])
        v.append(derivative)
    return v
