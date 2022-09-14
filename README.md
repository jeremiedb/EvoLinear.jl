# EvoLinear

ML library implementing linear boosting along L1/L2 regularization.
Currently supports mse (squared-error) and logistic (logloss) regression tasks.


```julia
using EvoLinear
config = EvoLinearRegressor(nrounds=10, loss=:mse, L1=1e-1, L2=10)
m = EvoLinear.fit(config; x, y, metric=:mse)
p = EvoLinear.predict_proj(m, x)
```

```julia
using EvoLinear
config = EvoLinearRegressor(nrounds=10, loss=:logistic, L1=1e-1, L2=10)
m = EvoLinear.fit(config; x, y, metric=:mse)
p = EvoLinear.predict_proj(m, x)
```