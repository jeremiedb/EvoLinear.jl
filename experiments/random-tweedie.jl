using Revise
using EvoLinear
using BenchmarkTools

nobs = 1_000_000
nfeats = 100
T = Float32

x_train = randn(T, nobs, nfeats)
coef = randn(T, nfeats) ./ 10

y_train = exp.(x_train * coef .+ rand(T, nobs) * T(0.1))
maximum(y_train)
mean(y_train)

config = EvoLinearRegressor(nrounds=10, loss=:tweedie, L1=0e-2, L2=0e-1)
@time m = EvoLinear.fit(config; x_train, y_train, metric=:tweedie_deviance)
sum(m.coef .== 0)

config = EvoLinearRegressor(nrounds=10, loss=:tweedie, L1=1e-2, L2=1e-1)
@btime EvoLinear.fit(config; x_train, y_train);

p = EvoLinear.predict_proj(m, x_train)
@time EvoLinear.tweedie_deviance(p, y_train)
@btime EvoLinear.tweedie_deviance($p, $y_train);


using XGBoost
# xgboost aprams
params_xgb = [
    "booster" => "gblinear",
    "updater" => "shotgun", # shotgun / coord_descent
    "eta" => 1.0,
    "objective" => "reg:tweedie",
    "print_every_n" => 5]

nthread = Threads.nthreads()
nthread = 8

nrounds = 100
metrics = ["mae"]
# metrics = ["gamma-deviance"]
# metrics = ["tweedie-nloglik@1.5"]

@info "xgboost train:"
@time m_xgb = xgboost(x, nrounds; label=y, param=params_xgb, metrics, nthread, silent=1);
