using CSV
using DataFrames
using EvoLinear
using XGBoost
using StatsBase: sample
using Random: seed!

using AWS: AWSCredentials, AWSConfig, @service
@service S3
aws_creds = AWSCredentials(ENV["AWS_ACCESS_KEY_ID_JDB"], ENV["AWS_SECRET_ACCESS_KEY_JDB"])
aws_config = AWSConfig(; creds=aws_creds, region="ca-central-1")

path = "share/data/aicrowd/insurance-aicrowd.csv"
raw = S3.get_object(
    "jeremiedb",
    path,
    Dict("response-content-type" => "application/octet-stream");
    aws_config,
)
df = DataFrame(CSV.File(raw))

target_name = "event"
feature_names = [
    "vh_age",
    "vh_value",
    "vh_speed",
    "vh_weight",
    "drv_age1",
    "pol_no_claims_discount",
    "pol_coverage",
    "pol_duration",
    "pol_sit_duration",
]

transform!(df, "claim_amount" => ByRow(x -> x > 0 ? 1.0f0 : 0.0f0) => "event")

pol_cov_dict = Dict{String,Float64}("Min" => 1, "Med1" => 2, "Med2" => 3, "Max" => 4)
pol_cov_map(x) = get(pol_cov_dict, x, 4)
transform!(df, "pol_coverage" => ByRow(pol_cov_map) => "pol_coverage")

seed!(123)
nobs = nrow(df)
id_train = sample(1:nobs, Int(round(0.8 * nobs)), replace=false)

dtrain = dropmissing(df[id_train, [feature_names..., target_name]])
deval = dropmissing(df[Not(id_train), [feature_names..., target_name]])

config = EvoLinearRegressor(
    loss=:logistic,
    L1=0.0,
    L2=0.0,
    nrounds=1000,
    eta=0.5,
)

# @time m = fit_evotree(config; x_train, y_train, print_every_n=25);
@time m = EvoLinear.fit(config, dtrain;
    feature_names,
    target_name,
    deval,
    metric=:logloss,
    early_stopping_rounds=100,
    print_every_n=10,
);

p_linear = m(deval);
EvoLinear.Metrics.logloss(p_linear, deval[:, target_name])

config = EvoSplineRegressor(
    T=Float32,
    loss=:logistic,
    nrounds=600,
    eta=1e-3,
    knots=Dict(1 => 4, 2 => 4, 3 => 4, 4 => 4, 5 => 4, 6 => 4, 7 => 4, 8 => 4, 9 => 4),
    act=:elu,
    batchsize=4096,
    device=:cpu,
)
@time m, logger = EvoLinear.fit(
    config;
    x_train,
    y_train,
    x_eval,
    y_eval,
    early_stopping_rounds=50,
    print_every_n=10,
    metric=:logloss,
    return_logger=true,
);
# @time m = EvoLinear.fit(config; x_train, y_train);
p_spline = m(x_eval')
# p_spline = m(x_eval' |> EvoLinear.Splines.gpu) |> EvoLinear.Splines.cpu
EvoLinear.Metrics.logloss(p_spline, y_eval)


######################################################
# XGBoost
######################################################
x_train = Matrix(dtrain[:, feature_names])
y_train = dtrain[:, target_name]

x_eval = Matrix(deval[:, feature_names])
y_eval = deval[:, target_name]

params_xgb = Dict(
    :objective => "reg:logistic",
    :booster => "gbtree",
    :eta => 0.05,
    :max_depth => 4,
    :lambda => 1.0,
    :gamma => 0.0,
    :subsample => 0.5,
    :colsample_bytree => 0.8,
    :tree_method => "hist",
    :max_bin => 32,
    :print_every_n => 5,
)

nthread = Threads.nthreads()
nthread = 8

num_round = 250
metric_xgb = "logloss"

@info "xgboost train:"
watchlist = Dict("eval" => DMatrix(x_eval, y_eval))
@time m_xgb = xgboost(
    (x_train, y_train);
    watchlist,
    num_round,
    nthread,
    verbosity=0,
    eval_metric=metric_xgb,
    early_stopping_rounds=10,
    params_xgb...,
);
p_xgb_tree = XGBoost.predict(m_xgb, x_eval)

params_xgb = Dict(
    :booster => "gblinear",
    :updater => "shotgun", # shotgun / coord_descent
    :eta => 1.0,
    :lambda => 0.0,
    :objective => "reg:logistic",
    :print_every_n => 5,
)

@info "xgboost train:"
@time m_xgb = xgboost(
    (x_train, y_train);
    watchlist,
    num_round,
    eval_metric=metric_xgb,
    early_stopping_rounds=10,
    nthread,
    verbosity=0,
    params_xgb...,
);
p_xgb_linear = XGBoost.predict(m_xgb, x_eval)

EvoLinear.Metrics.logloss(p_linear, y_eval)
EvoLinear.Metrics.logloss(p_spline, y_eval)
EvoLinear.Metrics.logloss(p_xgb_tree, y_eval)
EvoLinear.Metrics.logloss(p_xgb_linear, y_eval)