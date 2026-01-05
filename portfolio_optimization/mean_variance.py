import numpy as np

def portfolio_return(weights, mean_returns):
    return np.dot(weights, mean_returns)

def portfolio_volatility(weights, cov_matrix):
    weights = np.asarray(weights)
    cov_matrix = np.asarray(cov_matrix)
    return np.sqrt(weights @ cov_matrix @ weights)

def sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0.0425):
    p_return = portfolio_return(weights, mean_returns)
    p_volatility = portfolio_volatility(weights, cov_matrix)
    return (p_return - risk_free_rate) / p_volatility