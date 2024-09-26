// Define constants and functions
const S_0 = 100/* value of S_0 */;

function normPdf(x, mean, variance) {
    const scale = Math.sqrt(variance);
    const expTerm = Math.exp(-0.5 * Math.pow((x - mean) / scale, 2));
    return expTerm / (scale * Math.sqrt(2 * Math.PI));
}

function B(s) {
    if (s === 0) return 0; // Avoid division by zero
    return normPdf(s, 0, 1 / Math.max(s, 0.01)); // Safeguard small s values
}

function u(t, r) {
    // Define the function u(t, r)
    return r*Math.log(1+t)/* expression for u(t, r) */;
}

function s(t) {
    // Define the function s(t)
    return Math.log(1+t) /* expression for s(t) */;
}

function g(t, r) {
    // Define the function g(t, r)
    return Math.abs(Math.log(f(t))*(1-t/f(t)))+Math.log2(t)/* expression for g(t, r) */;
}

function k(t){
    return t**3-3*t**2*(1-t)
}


function v1(t) {
    // Define the function v1(t)
    return (Math.log(t)-Math.log(t/k(t)))*Math.log(1+g(t));
}

const d_t = 0.002739726027;

// Numerical integration using Trapezoidal rule
function integrate(f, a, b, n) {
    let h = (b - a) / n;
    let sum = 0.5 * (f(a) + f(b));
    for (let i = 1; i < n; i++) {
        sum += f(a + i * h);
    }
    return sum * h;
}

// Function for the integral
function integrand(x, t, r) {
    return x ** 2 * B(x) ** 2 * (Math.exp(u(t, r) * d_t + s(t) * B(t) * g(t, r)) + v1(t));
}

// Define p3(t, r) function
function p3(t, r) {
    const integral = integrate((x) => integrand(x, t, r), 0, t, 1000); // 1000 is the number of intervals for approximation
    return S_0 * (1 + integral);
}

// Example usage:
const t = 1/* some value for t */;
const r = 1 /* some value for r */;
console.log(p3(t, r));