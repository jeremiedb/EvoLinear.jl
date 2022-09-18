# EvoLinear

| Documentation | CI Status |
|:------------------------:|:----------------:|
| [![][docs-stable-img]][docs-stable-url] [![][docs-latest-img]][docs-latest-url] | [![][ci-img]][ci-url] |

[docs-latest-img]: https://img.shields.io/badge/docs-latest-blue.svg
[docs-latest-url]: https://jeremiedb.github.io/EvoLinear.jl/dev

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://jeremiedb.github.io/EvoLinear.jl/stable

[ci-img]: https://github.com/jeremiedb/EvoLinear.jl/workflows/CI/badge.svg
[ci-url]: https://github.com/jeremiedb/EvoLinear.jl/actions?query=workflow%3ACI+branch%3Amain


ML library implementing linear boosting with L1 and L2 regularization.
For Tree based boosting, consider [EvoTrees.jl](https://github.com/Evovest/EvoTrees.jl).

Supported loss functions:

- mse (squared-error)
- logistic (logloss) regression
- poisson
- gamma
- tweedie

## Installation

From General Registry

```julia
pkg> add EvoLinear
```

For latest version

```julia
pkg> add https://github.com/jeremiedb/EvoLinear.jl
```

## Getting started

Build a configuration struct with `EvoLinearRegressor`. Then `EvoLinear.fit` takes `x::Matrix` and `y::Vector` as inputs, plus optionally `w::Vector` as weights and fits a linear boosted model.

```julia
using EvoLinear
config = EvoLinearRegressor(loss=:mse, nrounds=10, L1=1e-1, L2=1e-2)
m = EvoLinear.fit(config; x, y, metric=:mse)
p = EvoLinear.predict_proj(m, x)
p = m(x)
```
