using Revise
using EvoLinear
using BenchmarkTools

nobs = 1_000_000
nfeats = 100
T = Float32

x = randn(T, nobs, nfeats)
coef = randn(T, nfeats) ./ 4

y = exp.(x * coef .+ rand(T, nobs) * T(0.1))
maximum(y)
mean(y)

config = EvoLinearRegressor(nrounds=10, loss=:gamma, L1=0e-2, L2=0e-1)
@time m = EvoLinear.fit(config; x, y, metric=:gamma)
sum(m.coef .== 0)

config = EvoLinearRegressor(nrounds=10, loss=:gamma, L1=1e-2, L2=1e-1)
@btime m = EvoLinear.fit(config; x, y, metric=:gamma);

p = EvoLinear.predict_proj(m, x)
@time EvoLinear.gamma(p, y)
@btime EvoLinear.gamma($p, $y);


using XGBoost
# xgboost aprams
params_xgb = [
    "booster" => "gblinear",
    "updater" => "shotgun", # shotgun / coord_descent
    "eta" => 1.0,
    "objective" => "reg:gamma",
    "print_every_n" => 5]

nthread = Threads.nthreads()
nthread = 8

nrounds = 10

metrics = ["gamma-deviance"]

@info "xgboost train:"
@time m_xgb = xgboost(x, nrounds, label=y, param=params_xgb, metrics=metrics, nthread=nthread, silent=0);
