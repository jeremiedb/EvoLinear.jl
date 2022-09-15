seed!(121)

nobs = 100_000
nfeats = 10
T = Float32

x = randn(T, nobs, nfeats)
coef = randn(T, nfeats)

y = x * coef .+ rand(T, nobs) * T(0.1)

config = EvoLinear.EvoLinearRegressor(nrounds=10, loss=:mse, L1=1, L2=1e-2)
m = EvoLinear.fit(config; x, y, metric=:mse)

@test sum(m.coef .== 0) > 2

m, cache = EvoLinear.init(config; x, y)
EvoLinear.fit!(m, cache, config);
p = EvoLinear.predict_proj(m, x)
metric = EvoLinear.mse(p, y)
metric = EvoLinear.mae(p, y)
@test metric < 2