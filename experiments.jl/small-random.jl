using Revise
using EvoLinear
using BenchmarkTools

nobs = 1_000_000
T = Float32

x1 = rand(T, nobs)
x2 = rand(T, nobs)

β1 = 2
β2 = 3

x = hcat(x1, x2)
y = β1 * x1 + β2 * x2 + rand(T, nobs) * T(0.001)

m = EvoLinear.fit(config; x, y)
m

# EvoLinear.predict(m, x)
m, cache = EvoLinear.init(config, x)
@time EvoLinear.fit!(m, cache, config; x, y)
# @code_warntype EvoLinear.fit!(m, cache, config; x, y)
# @btime EvoLinear.fit!($m, $cache, $config; x=$x, y=$y)
@info m

p = EvoLinear.predict(m, x)
metric = mse(p, y)
@info metric