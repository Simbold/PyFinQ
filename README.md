# PyFinQ
Basic derivative pricing in Python
    
- Fast pricing functions (utilisation of the Numba JIT compiler)
- Dependencies: Numpy, SciPy and Numba
- Currently implemented pricing functions:
        
    - Binomial tree model of Cox Ross and Rubinstein (American/European exercise style)
    - Finite-differences techniques for European exercise style call in the Black-Scholes model (Explicit, Implicit and Crank-Nicolson) 
    - Finite-differences scheme for American exercise style put in the Black-Scholes model (Crank-Nicolson scheme using the Brennan-Schwartz algorithm)
    - Explicit Black-Scholes formula (European call/put)
    - Heston model (European call/put) via the Laplace transform approach
    - Heston and Black-Scholes model (European call/put) via the Fast Fourier transform