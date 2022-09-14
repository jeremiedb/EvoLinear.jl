using Revise
using EvoLinear
using BenchmarkTools

nobs = 1_000_000
nfeats = 100
T = Float32

x = randn(T, nobs, nfeats)
# x = randn(T, nobs, nfeats) .+ 5 .* rand(T, nobs, nfeats)
coef = randn(T, nfeats)

y =  x * coef .+ rand(T, nobs) * T(0.01)

config = EvoLinear.EvoLinearRegressor(nrounds=10, loss=:mse, updater=:all)
@time m = EvoLinear.fit(config; x, y)
m

# EvoLinear.predict(m, x)
m, cache = EvoLinear.init(config, x)
@time EvoLinear.fit!(m, cache, config; x, y)
# @code_warntype EvoLinear.fit!(m, cache, config; x, y)
# @btime EvoLinear.fit!($m, $cache, $config; x=$x, y=$y)
@info m
p = EvoLinear.predict_proj(m, x)
metric = EvoLinear.mse(p, y)
metric = EvoLinear.mae(p, y)
@info metric


using XGBoost
# xgboost aprams
params_xgb = [
    "booster" => "gblinear",
    "updater" => "shotgun", # shotgun / coord_descent
    "eta" => 1.0,
    "objective" => "reg:squarederror",
    "print_every_n" => 5]

nthread = Threads.nthreads()
nrounds = 20

# metrics = ["rmse"]
metrics = ["mae"]
# metrics = ["logloss"]

@info "xgboost train:"
@time m_xgb = xgboost(x, nrounds, label=y, param=params_xgb, metrics=metrics, nthread=nthread, silent=0);
