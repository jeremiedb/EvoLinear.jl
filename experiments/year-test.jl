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
df = DataFrame(CSV.File(raw, header = false))

path = "share/data/year/year-train-idx.txt"
raw = S3.get_object(
    "jeremiedb",
    path,
    Dict("response-content-type" => "application/octet-stream");
    aws_config,
)
train_idx = DataFrame(CSV.File(raw, header = false))[:, 1] .+ 1

path = "share/data/year/year-eval-idx.txt"
raw = S3.get_object(
    "jeremiedb",
    path,
    Dict("response-content-type" => "application/octet-stream");
    aws_config,
)
eval_idx = DataFrame(CSV.File(raw, header = false))[:, 1] .+ 1

X = df[:, 2:end]
Y_raw = Float64.(df[:, 1])
Y = (Y_raw .- mean(Y_raw)) ./ std(Y_raw)

function percent_rank(x::AbstractVector{T}) where {T}
    return tiedrank(x) / (length(x) + 1)
end

transform!(X, names(X) .=> percent_rank .=> names(X))
X = collect(Matrix{Float32}(X))
Y = Float32.(Y)

x_tot, y_tot = X[1:(end-51630), :], Y[1:(end-51630)]
x_test, y_test = X[(end-51630+1):end, :], Y[(end-51630+1):end]
x_train, x_eval = x_tot[train_idx, :], x_tot[eval_idx, :]
y_train, y_eval = y_tot[train_idx], y_tot[eval_idx]


config = EvoLinearRegressor(
    T = Float32,
    loss = :mse,
    L1 = 0.0,
    L2 = 0.0,
    nrounds = 300,
    eta = 0.5,
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
    metric = :mse,
    return_logger = true,
);
@time p_linear = m(x_test);
mean((p_linear .- y_test) .^ 2) * std(Y_raw)^2

config = EvoSplineRegressor(
    T = Float32,
    loss = :mse,
    nrounds = 32,
    eta = 1e-2,
    knots = Dict(
        1 => 8,
        3 => 8,
        2 => 8,
        6 => 8,
        14 => 8,
        20 => 8,
        13 => 4,
        57 => 4,
        36 => 4,
        38 => 4,
    ),
    act = :tanh,
    batchsize = 4096,
    device = :cpu,
)
@time m, logger = EvoLinear.fit(
    config;
    x_train,
    y_train,
    x_eval,
    y_eval,
    early_stopping_rounds = 100,
    print_every_n = 10,
    metric = :mse,
    return_logger = true,
);
@time p_spline = m(x_test');
# @profview m(x_test');

# p_spline = m(x_test' |> EvoLinear.Splines.gpu) |> EvoLinear.Splines.cpu;
mean((p_spline .- y_test) .^ 2) * std(Y_raw)^2

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