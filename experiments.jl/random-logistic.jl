using Revise
using EvoLinear
using BenchmarkTools

nobs = 1_000_000
nfeats = 10
T = Float32

x = randn(T, nobs, nfeats)
coef = randn(T, nfeats)

y =  EvoLinear.sigmoid(x * coef) #.+ rand(T, nobs) * T(0.0001))

config = EvoLinear.EvoLinearRegressor(loss=:logistic)
@time m = EvoLinear.fit(config; x, y)
m

# EvoLinear.predict(m, x)
m, cache = EvoLinear.init(config, x)
@time EvoLinear.fit!(m, cache, config; x, y);
# @code_warntype EvoLinear.fit!(m, cache, config; x, y)
# @btime EvoLinear.fit!($m, $cache, $config; x=$x, y=$y)
@info m
p = EvoLinear.predict(m, x)
y_logit = EvoLinear.logit(y)
metric = EvoLinear.mse(p, y)
@info metric