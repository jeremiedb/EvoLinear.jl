using Revise
using EvoLinear
using BenchmarkTools

nobs = 1_000_000
nfeats = 100
T = Float32

x = randn(T, nobs, nfeats)
coef = randn(T, nfeats)

y = x * coef .+ rand(T, nobs) * T(0.1)

config = EvoLinear.EvoLinearRegressor(nrounds=10, loss=:mse, L1=0e-1, L2=1)
@time m, logger = EvoLinear.fit(config; x, y, metric_name=:mae, x_eval = x, y_eval = y, print_every_n = 20);
@btime m, logger = EvoLinear.fit(config; x, y, metric_name=:mae, x_eval = x, y_eval = y, print_every_n = 20);
sum(m.coef .== 0)

config = EvoLinear.EvoLinearRegressor(nrounds=10, loss=:mse, L1=1e-1, L2=1e-2)
@btime m = EvoLinear.fit(config; x, y);

@time m0, cache = EvoLinear.init(config; x, y)
@time EvoLinear.fit!(m0, cache, config);
# @btime EvoLinear.fit!($m, $cache, $config; x=$x, y=$y)
@code_warntype EvoLinear.fit!(m0, cache, config)
cache[:logger][:nrounds]

p = EvoLinear.predict_proj(m, x)
@btime m($x);

@btime metric = EvoLinear.mse(p, cache.y)
@btime metric = EvoLinear.mse(p, cache.y, cache.w)

metric = EvoLinear.mse(p, y)
metric = EvoLinear.mae(p, y)

@info metric

using XGBoost
# xgboost aprams
params_xgb = [
    "booster" => "gblinear",
    "updater" => "shotgun", # shotgun / coord_descent
    "eta" => 1.0,
    "lambda" => 0.0,
    "objective" => "reg:squarederror",
    "print_every_n" => 5]

nthread = Threads.nthreads()
nrounds = 10

# metrics = ["rmse"]
metrics = ["mae"]
# metrics = ["logloss"]

@info "xgboost train:"
@time m_xgb = xgboost(x, nrounds, label=y, param=params_xgb, metrics=metrics, nthread=nthread, silent=0);
