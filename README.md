# EvoLinear

| Documentation | CI Status | Coverage |
|:------------------------:|:----------------:|:----------------:|
| [![][docs-stable-img]][docs-stable-url] [![][docs-latest-img]][docs-latest-url] | [![][ci-img]][ci-url] | [![][cov-img]][cov-url] |

[docs-latest-img]: https://img.shields.io/badge/docs-latest-blue.svg
[docs-latest-url]: https://jeremiedb.github.io/EvoLinear.jl/dev

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://jeremiedb.github.io/EvoLinear.jl/stable

[ci-img]: https://github.com/jeremiedb/EvoLinear.jl/workflows/CI/badge.svg
[ci-url]: https://github.com/jeremiedb/EvoLinear.jl/actions?query=workflow%3ACI+branch%3Amain

[cov-img]: https://codecov.io/github/jeremiedb/evolinear.jl/branch/main/graph/badge.svg
[cov-url]: https://app.codecov.io/github/jeremiedb/evolinear.jl

ML library implementing linear boosting with L1 and L2 regularization.
For tree based boosting, consider [EvoTrees.jl](https://github.com/Evovest/EvoTrees.jl).

Supported loss functions:

- `mse`: mean squared-error regression
- `logloss`: logistic regression
- `poisson`
- `gamma`:
- `tweedie`:

## Installation

From General Registry

```
pkg> add EvoLinear
```

For latest version

```
pkg> add https://github.com/jeremiedb/EvoLinear.jl
```

## Getting started

Define a learner with `EvoLinearRegressor`. This objects holds the hyper-paramters of the model. 

Then `EvoLinear.fit` trains a model defined in the learner on a `Tables` compatible objects. The features, target and optionally weight variable names must be specified. 

```julia
using EvoLinear, DataFrames
using EvoLinear: fit

x_train, y_train = rand(1_000, 10), rand(1_000)
dtrain = DataFrame(x_train, :auto)
dtrain.y .= y_train

config = EvoLinearRegressor(loss=:mse, nrounds=10, L1=1e-1, L2=1e-2)
m = fit(config, dtrain; target_name="y", feature_names=["x1", "x3"]);
p = m(dtrain)
```
