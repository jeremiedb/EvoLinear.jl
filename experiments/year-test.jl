using Revise
using CSV
using DataFrames
using EvoLinear
using XGBoost
using StatsBase: sample, tiedrank
using Statistics
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

path = "share/data/year/year-train-idx.txt"
raw = S3.get_object(
    "jeremiedb",
    path,
    Dict("response-content-type" => "application/octet-stream");
    aws_config,
)
train_idx = DataFrame(CSV.File(raw, header=false))[:, 1] .+ 1

path = "share/data/year/year-eval-idx.txt"
raw = S3.get_object(
    "jeremiedb",
    path,
    Dict("response-content-type" => "application/octet-stream");
    aws_config,
)
eval_idx = DataFrame(CSV.File(raw, header=false))[:, 1] .+ 1

X = df[:, 2:end]
Y_raw = Float64.(df[:, 1])
Y = (Y_raw .- mean(Y_raw)) ./ std(Y_raw)

function percent_rank(x::AbstractVector{T}) where {T}
    return tiedrank(x) / (length(x) + 1)
end

transform!(X, names(X) .=> percent_rank .=> names(X))
X = collect(Matrix{Float32}(X)')
Y = Float32.(Y)

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