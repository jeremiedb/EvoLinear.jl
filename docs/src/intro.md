# EvoLinear.jl

ML library implementing linear boosting with L1 and L2 regularization.

For tree based boosting, consider [EvoTrees.jl](https://github.com/Evovest/EvoTrees.jl).

Supported loss functions:

- `mse`: mean squared-error regression
- `logloss`: logistic regression
- `poisson`
- `gamma`:
- `tweedie`:

## Installation

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
