import numpy as np
import cvxpy as cp

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

def optimise_portfolio(mean_returns, cov_matrix, target_return):
    mean_returns = np.asarray(mean_returns)
    cov_matrix = np.asarray(cov_matrix)

    #define the unknown variable x : f(x) . THis creates a vector of variables w = (w1, w2, ..., wn)
    weights = cp.Variable(len(mean_returns))


    # objective - find the portfolio weights that minimize portfolio variance


    objective = cp.Minimize(cp.quad_form(weights, cov_matrix)) 

    constraints = [
    cp.sum(weights) == 1,
    mean_returns @ weights >= target_return,
    weights >= 0
    ]

    problem = cp.Problem(objective, constraints)
    problem.solve()

    return np.round(weights.value, 6)