# EvoLinear.jl

ML library implementing linear boosting with L1/L2 regularization.
For Tree based boosting, consider [EvoTrees.jl](https://github.com/Evovest/EvoTrees.jl).

Currently supports:    
- mse (squared-error)
- logistic (logloss) regression
- Poisson
- Gamma
- Tweedie

## Installation

```
pkg> add https://github.com/jeremiedb/EvoLinear.jl
```

## Getting started

`EvoLinear.fit` takes `x::Matrix` and `y::Vector` as inputs, plus optionally `w::Vector` as weights.

```julia
using EvoLinear
config = EvoLinearRegressor(nrounds=10, loss=:mse, L1=1e-1, L2=1e-2)
m = EvoLinear.fit(config; x, y, metric=:mse)
p = EvoLinear.predict_proj(m, x)
```

```julia
using EvoLinear
config = EvoLinearRegressor(nrounds=10, loss=:logistic, L1=1e-1, L2=1e-2)
m = EvoLinear.fit(config; x, y, metric=:mse)
p = EvoLinear.predict_proj(m, x)
```