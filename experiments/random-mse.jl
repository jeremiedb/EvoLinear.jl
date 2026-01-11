using DataFrames
using EvoLinear
using BenchmarkTools

nobs = 1_000_000
nfeats = 100
T = Float32
train_pct = 0.8

x = randn(T, nobs, nfeats)
coef = randn(T, nfeats)
y = x * coef .+ rand(T, nobs) * T(0.1)

dtot = DataFrame(x, :auto)
dtot.y = y
target_name = :y
feature_names = setdiff(Symbol.(names(dtot)), [target_name])

train_idx = 1:round(Int, train_pct * nobs)
eval_idx = setdiff(1:nobs, train_idx)
dtrain = dtot[train_idx, :]
deval = dtot[eval_idx, :]

config = EvoLinearRegressor(nrounds=100, loss=:mse, eta=0.3, L1=0e-1, L2=1, early_stopping_rounds=10)
# @time m = EvoLinear.fit(config, dtrain; target_name, feature_names);
@time m = EvoLinear.fit(config, dtrain; target_name, feature_names, deval=dtrain, print_every_n=5);
# @time m, logger = EvoLinear.fit(config; x_train, y_train, metric=:mae, x_eval = x_train, y_eval = y_train, print_every_n = 5, return_logger = true);
# @btime m, logger = EvoLinear.fit(config; x_train, y_train, metric=:mae, x_eval = x_train, y_eval = y_train, print_every_n = 5, return_logger = true);
sum(m.coef .== 0)
m.info[:logger][:nrounds]

config = EvoLinear.EvoLinearRegressor(nrounds=10, loss=:mse, L1=1e-1, L2=1e-2)
# @btime m = EvoLinear.fit(config; x_train, y_train);

@time m0, cache = EvoLinear.init(config, dtrain; feature_names, target_name)
@time EvoLinear.fit!(m0, cache, config);
# @btime EvoLinear.fit!($m, $cache, $config; x=$x, y=$y)
@code_warntype EvoLinear.fit!(m0, cache, config)

p = EvoLinear.predict(m, dtrain)
@time m(dtrain);
@btime m($dtrain);

@btime metric = EvoLinear.Metrics.mse(p, cache.y)
@btime metric = EvoLinear.Metrics.mse(p, cache.y, cache.w)
metric = EvoLinear.Metrics.mse(p, dtrain[!, target_name])


# #####################################
# # XGBoost
# #####################################
# @info "xgboost train"
# using XGBoost

# x_train = dtrain[:, feature_names]
# y_train = dtrain[:, target_name]

# # xgboost aprams
# params_xgb = [
#     :booster => "gblinear",
#     :updater => "shotgun", # shotgun / coord_descent
#     :eta => 0.1,
#     :lambda => 0.0,
#     :num_round => nrounds,
#     :objective => "reg:squarederror",
#     :print_every_n => 5]

# nthread = Threads.nthreads()

# # metrics = ["rmse"]
# metric_xgb = ["mae"]
# # metrics = ["logloss"]

# dtrain_xgb = DMatrix(x_train, y_train)
# watchlist = Dict("train" => DMatrix(x_train, y_train))
# @time m_xgb = xgboost(dtrain_xgb; watchlist, nthread, verbosity=0, silent=0, eval_metric=metric_xgb, params_xgb...)
