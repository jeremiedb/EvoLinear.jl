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

config = EvoLinearRegressor(nrounds=10, loss=:poisson, L1=0e-2, L2=0e-1)
@time m = EvoLinear.fit(config; x_train, y_train, metric=:poisson_deviance)
sum(m.coef .== 0)

config = EvoLinearRegressor(nrounds=10, loss=:poisson, L1=1e-2, L2=1e-1)
@btime m = EvoLinear.fit(config; x_train, y_train);

p = EvoLinear.predict_proj(m, x_train)
@time EvoLinear.poisson_deviance(p, y_train)


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
