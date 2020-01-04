# PyFinQ
Basic derivative pricing in Python
    
- Fast pricing functions (utilisation of the Numba JIT compiler)
- Dependencies: Numpy, SciPy and Numba
- Currently implemented pricing functions:
        
    - Binomial tree model of Cox Ross and Rubinstein (American/European exercise style)
    - Finite-differences techniques for European exercise style in the Black-Scholes model (Explicit, Implicit and Crank-Nicolson) 
    - Finite-differences scheme for American exercise style in the Black-Scholes model (Crank-Nicolson scheme using the Brennan-Schwartz algorithm)
    - Black-Scholes explicit formula (for European put/call)
    - European Call/Put in the Heston model via the Laplace transform approach