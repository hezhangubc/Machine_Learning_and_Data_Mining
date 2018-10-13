def predict(X)
    if X[0] < -80.305:
        if X[1] < 37.67:
            y = 1
        else:
            y = 0            
    return y
