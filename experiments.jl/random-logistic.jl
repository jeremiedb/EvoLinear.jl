using Revise
using EvoLinear
using BenchmarkTools

nobs = 1_000_000
nfeats = 100
T = Float32

x = randn(T, nobs, nfeats)
coef = randn(T, nfeats)

y = EvoLinear.sigmoid(x * coef .+ rand(T, nobs) * T(0.1))

config = EvoLinear.EvoLinearRegressor(nrounds=10, eta=1.0, loss=:logistic, L1=0e-2, L2=0e-2)
@time m = EvoLinear.fit(config; x, y, metric=:logloss)
sum(m.coef .== 0)

config = EvoLinearRegressor(nrounds=10, loss=:logistic, L1=1e-2, L2=1e-1)
@btime m = EvoLinear.fit(config; x, y, metric=:logloss);

# EvoLinear.predict(m, x)
@time m, cache = EvoLinear.init(config; x, y)
@time EvoLinear.fit!(m, cache, config);
@code_warntype EvoLinear.fit!(m, cache, config)
# all: 139.865 ms (522 allocations: 782.04 MiB)
# single: 597.298 ms (1213 allocations: 1.13 GiB)
# @btime EvoLinear.fit!($m, $cache, $config; x=$x, y=$y)
# @info m
p = EvoLinear.predict_proj(m, x)
p = m(x; proj=true)

@code_warntype EvoLinear.predict_proj(m, x)
@code_warntype m(x; proj=true)

@btime EvoLinear.predict_proj($m, $x);
@btime m($x; proj=true);

@btime metric = EvoLinear.logloss(p, cache.y)
@btime metric = EvoLinear.logloss(p, cache.y, cache.w)


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
metrics = ["logloss"]

@info "xgboost train:"
@time m_xgb = xgboost(x, nrounds, label=y, param=params_xgb, metrics=metrics, nthread=nthread, silent=0);
