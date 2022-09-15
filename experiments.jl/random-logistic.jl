using Revise
using EvoLinear
using BenchmarkTools

nobs = 1_000_000
nfeats = 100
T = Float32

x = randn(T, nobs, nfeats)
coef = randn(T, nfeats)

y = EvoLinear.sigmoid(x * coef .+ rand(T, nobs) * T(0.1))

config = EvoLinear.EvoLinearRegressor(nrounds=10, loss=:logistic, L1=5e-2, L2=1e-2)
@time m = EvoLinear.fit(config; x, y, metric=:logloss)
sum(m.coef .== 0)

config = EvoLinearRegressor(nrounds=10, loss=:logistic, L1=1e-2, L2=1e-1)
@btime m = EvoLinear.fit(config; x, y, metric=:logloss);

# EvoLinear.predict(m, x)
@time m, cache = EvoLinear.init(config; x, y)
@time EvoLinear.fit!(m, cache, config);
# @code_warntype EvoLinear.fit!(m, cache, config; x, y)
# all: 139.865 ms (522 allocations: 782.04 MiB)
# single: 597.298 ms (1213 allocations: 1.13 GiB)
# @btime EvoLinear.fit!($m, $cache, $config; x=$x, y=$y)
# @info m
p = EvoLinear.predict_proj(m, x)

# 12.310 ms (6 allocations: 7.63 MiB)
# @btime p1 = EvoLinear.predict_linear($m, $x);

# 885.100 μs (2 allocations: 3.81 MiB)
# @btime p1 = EvoLinear.sigmoid($p);

y_logit = EvoLinear.logit(y)
metric = EvoLinear.mse(p, y)
metric = EvoLinear.mae(p, y)
metric = EvoLinear.logloss(p, y)
@info metric


using XGBoost
# xgboost aprams
params_xgb = [
    "booster" => "gblinear",
    "updater" => "shotgun", # shotgun / coord_descent
    "eta" => 1.0,
    "objective" => "reg:logistic",
    "print_every_n" => 5]

nthread = Threads.nthreads()
nthread = 8

nrounds = 20

# metrics = ["rmse"]
# metrics = ["mae"]
metrics = ["logloss"]

@info "xgboost train:"
@time m_xgb = xgboost(x, nrounds, label=y, param=params_xgb, metrics=metrics, nthread=nthread, silent=0);
