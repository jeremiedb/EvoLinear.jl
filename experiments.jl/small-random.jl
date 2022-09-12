using Revise
using EvoLinear

nobs = 100_000

x1 = rand(nobs)
x2 = rand(nobs)

β1 = 2
β2 = 3

x = hcat(x1, x2)
y = β1 * x1 + β2 * x2 + rand(nobs) .* 0.1

config = EvoLinear.EvoLinearRegressor();
EvoLinear.fit(config, x, y)
