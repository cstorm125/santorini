def discounted_returns(returns, gamma = 0.5):
    G = 0
    for i in range(len(returns)):
        G+=returns[i] * gamma ** i
    return G