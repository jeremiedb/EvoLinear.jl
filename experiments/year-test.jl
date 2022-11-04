using Revise
using CSV
using DataFrames
using EvoLinear
using XGBoost
using StatsBase: sample
using Random: seed!

using AWS: AWSCredentials, AWSConfig, @service
@service S3
aws_creds = AWSCredentials(ENV["AWS_ACCESS_KEY_ID_JDB"], ENV["AWS_SECRET_ACCESS_KEY_JDB"])
aws_config = AWSConfig(; creds = aws_creds, region = "ca-central-1")

path = "share/data/year/year.csv"
raw = S3.get_object(
    "jeremiedb",
    path,
    Dict("response-content-type" => "application/octet-stream");
    aws_config,
)
df = DataFrame(CSV.File(raw, header=false))

seed!(123)
nobs = nrow(df)
id_train = sample(1:nobs, Int(round(0.8 * nobs)), replace = false)

df_train = dropmissing(df[id_train, [feats..., target]])
df_eval = dropmissing(df[Not(id_train), [feats..., target]])

x_train = Matrix{Float32}(df_train[:, feats])
x_eval = Matrix{Float32}(df_eval[:, feats])
y_train = Vector{Float32}(df_train[:, target])
y_eval = Vector{Float32}(df_eval[:, target])

config = EvoLinearRegressor(
    T = Float32,
    loss = :logistic,
    L1 = 0.0,
    L2 = 0.0,
    nrounds = 1000,
    eta = 0.2,
)

# @time m = fit_evotree(config; x_train, y_train, print_every_n=25);
@time m, logger = EvoLinear.fit(
    config;
    x_train,
    y_train,
    x_eval,
    y_eval,
    early_stopping_rounds = 100,
    print_every_n = 10,
    metric = :logloss,
    return_logger = true,
);
p_linear = m(x_eval);

config = EvoSplineRegressor(
    T = Float32,
    loss = :logistic,
    nrounds = 400,
    eta = 1e-3,
    knots = Dict(1 => 4, 2 => 4, 3 => 4, 4 => 4, 5 => 4, 6 => 4, 7 => 4, 8 => 4),
    act = :relu,
    batchsize = 2048,
    device = :cpu,
)

@time m, logger = EvoLinear.fit(
    config;
    x_train,
    y_train,
    x_eval,
    y_eval,
    early_stopping_rounds = 50,
    print_every_n = 10,
    metric = :logloss,
    return_logger = true,
);
# @time m = EvoLinear.fit(config; x_train, y_train);
p_spline = m(x_eval')

params_xgb = [
    "objective" => "reg:logistic",
    "booster" => "gbtree",
    "eta" => 0.05,
    "max_depth" => 4,
    "lambda" => 10.0,
    "gamma" => 0.0,
    "subsample" => 0.5,
    "colsample_bytree" => 0.8,
    "tree_method" => "hist",
    "max_bin" => 32,
    "print_every_n" => 5,
]

nthread = Threads.nthreads()
nthread = 8

nrounds = 250
metrics = ["logloss"]

@info "xgboost train:"
@time m_xgb = xgboost(
    x_train,
    nrounds,
    label = y_train,
    param = params_xgb,
    metrics = metrics,
    nthread = nthread,
    silent = 1,
);
p_xgb_tree = XGBoost.predict(m_xgb, x_eval)

params_xgb = [
    "booster" => "gblinear",
    "updater" => "shotgun", # shotgun / coord_descent
    "eta" => 1.0,
    "lambda" => 0.0,
    "objective" => "reg:logistic",
    "print_every_n" => 5,
]

nthread = Threads.nthreads()
nthread = 8

nrounds = 250
metrics = ["logloss"]

@info "xgboost train:"
@time m_xgb = xgboost(
    x_train,
    nrounds,
    label = y_train,
    param = params_xgb,
    metrics = metrics,
    nthread = nthread,
    silent = 1,
);
p_xgb_linear = XGBoost.predict(m_xgb, x_eval)

EvoLinear.Metrics.logloss(p_linear, y_eval)
EvoLinear.Metrics.logloss(p_spline, y_eval)
EvoLinear.Metrics.logloss(p_xgb_tree, y_eval)
EvoLinear.Metrics.logloss(p_xgb_linear, y_eval)