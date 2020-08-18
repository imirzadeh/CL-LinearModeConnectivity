def Bezier_connect(t, w1, w2, theta):
    return (1.0-t)**2 * w1 + 2*t*(1-t)*theta + t**2 * w2