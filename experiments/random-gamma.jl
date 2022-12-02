using Revise
using EvoLinear
using BenchmarkTools
using Random: seed!

seed!(123)
nobs = 1_000_000
nfeats = 100
T = Float32

x_train = randn(T, nobs, nfeats)
coef = randn(T, nfeats) ./ 5
bias = 1

y_train = exp.(x_train * coef .+ bias .+ rand(T, nobs) * T(0.1))
maximum(y_train)
mean(y_train)

config = EvoLinearRegressor(nrounds=10, loss=:gamma, L1=0e-2, L2=0e-1)
@time m = EvoLinear.fit(config; x_train, y_train, metric=:gamma_deviance)
sum(m.coef .== 0)

config = EvoLinearRegressor(nrounds=10, loss=:gamma, L1=1e-2, L2=1e-1)
@btime m = EvoLinear.fit(config; x_train, y_train, metric=:gamma_deviance);

p = EvoLinear.predict_proj(m, x_train)
@time EvoLinear.gamma_deviance(p, y_train)
@btime EvoLinear.gamma_deviance($p, $y_train);


using XGBoost
# xgboost aprams
params_xgb = [
    "booster" => "gblinear",
    "updater" => "shotgun", # shotgun / coord_descent
    "eta" => 1.0,
    "objective" => "reg:gamma",
    "print_every_n" => 5]

metrics = ["gamma-deviance"]
nthread = Threads.nthreads()
nthread = 8

nrounds = 10
@info "xgboost train:"
@time m_xgb = xgboost(x, nrounds, label=y, param=params_xgb, metrics=metrics, nthread=nthread, silent=0);
