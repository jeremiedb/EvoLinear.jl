# EvoLinear.jl

ML library implementing linear boosting with L1 and L2 regularization.

For tree based boosting, consider [EvoTrees.jl](https://github.com/Evovest/EvoTrees.jl).

Supported loss functions:

- mse (squared-error)
- logistic (logloss) regression
- poisson
- gamma
- tweedie

## Installation

```
pkg> add https://github.com/jeremiedb/EvoLinear.jl
```

## Getting started

Build a configuration struct with `EvoLinearRegressor`. Then `EvoLinear.fit` takes `x::Matrix` and `y::Vector` as inputs, plus optionally `w::Vector` as weights and fits a linear boosted model.

```julia
using EvoLinear
config = EvoLinearRegressor(loss=:mse, L1=1e-1, L2=1e-2, nrounds=10)
m = EvoLinear.fit(config; x, y, metric=:mse)
p = m(x)
```