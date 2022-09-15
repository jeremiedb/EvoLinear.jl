# EvoLinear

ML library implementing linear boosting along L1/L2 regularization.

Currently supports:
    
    - mse (squared-error)
    - logistic (logloss) regression
    - Poisson
    - Gamma
    - Tweedie

For Tree based boosting, consider [EvoTrees.jl](https://github.com/Evovest/EvoTrees.jl).

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