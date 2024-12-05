import numpy as np

const = 1/12

def sech(x):
    """
    Compute the hyperbolic secant of x.
    """
    return 1 / np.cosh(x)

def n1_func(x, alpha, r0, d, s0):
    """
    Compute n_1(x).

    Parameters:
    x : float : Input variable (e.g., time or intermediate value)
    alpha : float : Scaling constant for the sech function
    r0 : float : Initial interest rate
    d : float : Displacement constant
    s0 : float : Base value added to the function

    Returns:
    float : Value of n_1(x)
    """
    return alpha * sech(r0 * x - d) + s0



def B0_func(t, s, T, r0):
    """
    Compute B_0(t, s, T).

    Parameters:
    t : float : Current time
    s : float : Intermediate time
    T : float : Final time (maturity)
    r0 : float : Initial interest rate
    c : float : Constant scaling factor

    Returns:
    float : Value of B_0(t, s, T)
    """
    if s == t or T == t:
        raise ValueError("Denominator (s - t) or (T - t) cannot be zero.")
        
    numerator = r0 * np.exp((T - t) * r0 / const)
    denominator = (s - t) * (T - t)
    return numerator / denominator

def V3(t, s, T, C0, B0_func, K, u, v, rho, n1_func, r0,alpha,d,s0):
    """
    Compute the value of V_3(t, s, T).

    Parameters:
    t : float : Current time
    s : float : Intermediate time
    T : float : Final time (maturity)
    C0 : float : Initial constant (C0)
    B0_func : function : Function B0(t, s, T) that depends on t, s, and T
    K : float : Constant K
    u : float : Drift parameter
    v : float : Volatility parameter
    rho : float : Correlation coefficient
    n1_func : function : Function n1(s) that depends on s
    r0 : float : Initial interest rate

    Returns:
    float : Computed value of V_3(t, s, T)
    """

    # Compute B0(t, s, T)
    B0 = B0_func(t, s, T,r0)
    
    # Compute the exponent term
    exponent = (u - (v**2) / 2 + rho * n1_func(t,alpha,r0,d,s0)) * (T - t)
    exp_term = np.exp(exponent)
    
    # Compute the numerator
    numerator = (
        C0 +
        B0 * (1 - B0 / K) +
        exp_term +
        B0 * K * np.exp(-r0 * t)
    )
    
    # Compute the denominator
    exponent_denominator = ((s - t) * (T - s)) / (T - t)
    denominator = (1 + B0 + s) ** exponent_denominator
    
    # Compute V3
    V3_value = numerator / denominator
    return V3_value