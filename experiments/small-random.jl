using EvoLinear
using BenchmarkTools

nobs = 1_000_000
T = Float32

x1 = rand(T, nobs)
x2 = rand(T, nobs)

β1 = 2
β2 = 3

x_train = hcat(x1, x2)
y_train = β1 * x1 + β2 * x2 + rand(T, nobs) * T(0.001)

config = EvoLinear.EvoLinearRegressor(loss=:mse)
m = EvoLinear.fit(config; x_train, y_train)
m

# EvoLinear.predict(m, x)
m, cache = EvoLinear.init(config; x_train, y_train)
@time EvoLinear.fit!(m, cache, config)
# @code_warntype EvoLinear.fit!(m, cache, config; x_train, y_train)
@btime EvoLinear.fit!($m, $cache, $config)
@info m
p = EvoLinear.predict_proj(m, x_train)
metric = EvoLinear.mse(p, y_train)
@info metric