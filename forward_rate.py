import numpy as np
from scipy.integrate import quad

# Define sech(z) function as 1 / cosh(z)
def sech(z):
    return 1 / np.cosh(z)

# Define W(z), placeholder function (can be stochastic or Poisson process)
def W(z):
    return np.exp(-z)  # Example: Exponential decay

# Define B(t), can be a normal distribution (stochastic)
def B(t):
    # Here, we use a simple Gaussian-like function for B(t)
    return np.exp(-t**2)

# Function to calculate inner integrals
def inner_integrals(t, r):
    # First inner integral: ∫ B(x) dx from 0 to t
    integral_1, _ = quad(B, 0, t)
    
    # Second inner integral: ∫ B(x)^2 dx from 0 to t
    integral_2, _ = quad(lambda x: B(x)**2, 0, t)
    
    # Third outer integral: ∫(∫ B(x)^2 dx) dy from 0 to t
    outer_integral, _ = quad(lambda y: integral_2, 0, t)
    
    # Return the full expression inside the function
    return t**2 * B(t)**2 - 2 * t * r * B(t) * integral_1 + outer_integral

# Function f(t, r) using the integrals
def f(t, r):
    # Define the integrand for the outer integral
    def integrand(z):
        W_z = W(z)
        sech_z = sech(z)
        inner_expr = inner_integrals(t, r)
        return W_z * r**2 * sech_z * inner_expr

    # Perform the outer integration ∫ from 0 to t
    outer_integral, _ = quad(integrand, 0, t)
    return outer_integral

## Example usage
#t_val = 1.0  # Set the value of t
#r_val = 1.0  # Set the value of r
#
#result = f(t_val, r_val)
#print(f"f({t_val}, {r_val}) = {result}")