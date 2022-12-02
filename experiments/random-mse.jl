using Revise
using EvoLinear
using BenchmarkTools

nobs = 1_000_000
nfeats = 100
T = Float32

x_train = randn(T, nobs, nfeats)
coef = randn(T, nfeats)
y_train = x_train * coef .+ rand(T, nobs) * T(0.1)

config = EvoLinearRegressor(nrounds=10, loss=:mse, L1=0e-1, L2=1)
@time m, logger = EvoLinear.fit(config; x_train, y_train, metric=:mae, x_eval = x_train, y_eval = y_train, print_every_n = 5, return_logger = true);
# @btime m, logger = EvoLinear.fit(config; x_train, y_train, metric=:mae, x_eval = x_train, y_eval = y_train, print_every_n = 5, return_logger = true);
sum(m.coef .== 0)

config = EvoLinear.EvoLinearRegressor(nrounds=10, loss=:mse, L1=1e-1, L2=1e-2)
# @btime m = EvoLinear.fit(config; x_train, y_train);

@time m0, cache = EvoLinear.Linear.init(config, x_train, y_train)
@time EvoLinear.Linear.fit!(m0, cache, config);
# @btime EvoLinear.fit!($m, $cache, $config; x=$x, y=$y)
@code_warntype EvoLinear.Linear.fit!(m0, cache, config)
logger[:nrounds]

p = EvoLinear.Linear.predict_proj(m, x_train)
@btime m($x_train);

@btime metric = EvoLinear.Metrics.mse(p, cache.y)
@btime metric = EvoLinear.Metrics.mse(p, cache.y, cache.w)

metric = EvoLinear.Metrics.mse(p, y_train)
metric = EvoLinear.Metrics.mae(p, y_train)

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
@time m_xgb = xgboost(x_train, nrounds, label=y_train, param=params_xgb, metrics=metrics, nthread=nthread, silent=0);
