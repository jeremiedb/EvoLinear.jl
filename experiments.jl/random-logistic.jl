using Revise
using EvoLinear
using BenchmarkTools

using XGBoost

nobs = 1_000_000
nfeats = 100
T = Float32

x = randn(T, nobs, nfeats)
coef = randn(T, nfeats)

y = EvoLinear.sigmoid(x * coef .+ rand(T, nobs) * T(0.1))

config = EvoLinear.EvoLinearRegressor(loss=:logistic)
@time m = EvoLinear.fit(config; x, y)
m

# EvoLinear.predict(m, x)
@time m, cache = EvoLinear.init(config, x)
@time EvoLinear.fit!(m, cache, config; x, y);
# @code_warntype EvoLinear.fit!(m, cache, config; x, y)
# @btime EvoLinear.fit!($m, $cache, $config; x=$x, y=$y)
@info m
p = EvoLinear.sigmoid(EvoLinear.predict(m, x))
y_logit = EvoLinear.logit(y)
metric = EvoLinear.mse(p, y)
metric = EvoLinear.mae(p, y)
metric = EvoLinear.logloss(p, y)
@info metric


# xgboost aprams
params_xgb = [
    "booster" => "gblinear",
    "eta" => 1.0,
    "objective" => "reg:logistic",
    "print_every_n" => 5,
    "subsample" => 1.0,
    "colsample_bytree" => 1.0]

nthread = Threads.nthreads()
nrounds = 20

# metrics = ["rmse"]
metrics = ["mae"]
# metrics = ["logloss"]

@info "xgboost train:"
@time m_xgb = xgboost(x, nrounds, label=y, param=params_xgb, metrics=metrics, nthread=nthread, silent=0);
