# EvoLinear

ML library implementing linear boosting with L1 and L2 regularization.

For Tree based boosting, consider [EvoTrees.jl](https://github.com/Evovest/EvoTrees.jl).

<br>

| Documentation | CI Status |
|:------------------------:|:----------------:|
| [![][docs-stable-img]][docs-stable-url] [![][docs-latest-img]][docs-latest-url] | [![][ci-img]][ci-url] |

[docs-latest-img]: https://img.shields.io/badge/docs-latest-blue.svg
[docs-latest-url]: https://jeremiedb.github.io/EvoLinear.jl/dev

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://jeremiedb.github.io/EvoLinear.jl/stable

[ci-img]: https://github.com/jeremiedb/EvoLinear.jl/workflows/CI/badge.svg
[ci-url]: https://github.com/jeremiedb/EvoLinear.jl/actions?query=workflow%3ACI+branch%3Amain

<br>

Supported loss functions:

- mse (squared-error)
- logistic (logloss) regression
- poisson
- gamma
- tweedie

`EvoLinear.fit` takes `x::Matrix` and `y::Vector` as inputs, plus optionally `w::Vector` as weights.

```julia
using EvoLinear
config = EvoLinearRegressor(loss=:mse, L1=1e-1, L2=1e-2, nrounds=10)
m = EvoLinear.fit(config; x, y, metric=:mse)
p = EvoLinear.predict_proj(m, x)
```
