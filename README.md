# EvoLinear

> ML library implementing linear boosting with L1/L2 regularization.

| Documentation | CI Status | Coverage |
|:------------------------:|:----------------:|:----------------:|
| [![][docs-stable-img]][docs-stable-url] [![][docs-latest-img]][docs-latest-url] | [![][ci-img]][ci-url] | [![][codecov-img]][codecov-url] |

[docs-latest-img]: https://img.shields.io/badge/docs-latest-blue.svg
[docs-latest-url]: https://jeremiedb.github.io/EvoLinear.jl/dev

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://jeremiedb.github.io/EvoLinear.jl/stable

[ci-img]: https://github.com/jeremiedb/EvoLinear.jl/workflows/CI/badge.svg
[ci-url]: https://github.com/jeremiedb/EvoLinear.jl/actions?query=workflow%3ACI+branch%3Amain

[codecov-img]: https://codecov.io/gh/jeremiedb/EvoLinear.jl/coverage.svg?branch=main
[codecov-url]: https://codecov.io/gh/jeremiedb/EvoLinear.jl?branch=main


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