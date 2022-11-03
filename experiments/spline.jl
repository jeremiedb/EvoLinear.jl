using Revise
using EvoLinear
using AlgebraOfGraphics, GLMakie
using DataFrames
using BenchmarkTools

nobs = 1_000
nfeats = 1
T = Float32

x_train = rand(T, nobs, nfeats) .* 2
coef = randn(T, nfeats)

y_train = sin.(x_train[:, 1]) .+ randn(T, nobs) .* 0.1f0
df = DataFrame(hcat(x_train, y_train), ["x", "y"]);
draw(data(df) * mapping(:x, :y) * visual(Scatter, markersize = 5, color = "gray"))

config = EvoLinearRegressor(nrounds = 10, loss = :mse, L1 = 0e-1, L2 = 1)
@time ml = EvoLinear.fit(
    config;
    x_train,
    y_train,
    x_eval = x_train,
    y_eval = y_train,
    metric = :mae,
    print_every_n = 10,
);

x_pred =
    reshape(range(start = minimum(x_train), stop = maximum(x_train), length = 100), :, 1)
p = EvoLinear.Linear.predict_proj(ml, x_pred)

dfp = DataFrame(hcat(x_pred, p), ["x", "p"]);
plt =
    data(df) * mapping(:x, :y) * visual(Scatter, markersize = 5, color = "gray") +
    data(dfp) * mapping(:x, :p) * visual(Lines, markersize = 5, color = "navy")
draw(plt)

x1 = rand(100, 1)
@time ml(x1);

config = EvoSplineRegressor(
    loss = :mse,
    nrounds = 100,
    knots = Dict(1 => 16),
    act = :elu,
    eta = 1e-2,
    batchsize = 200,
)
m = EvoLinear.fit(config; x_train, y_train)
# fit!(loss, m, dtrain, opts)
p1 = ml(x_pred)
p2 = m(x_pred')
dfp = DataFrame(hcat(x_pred, p1, p2), ["x", "p_linear", "p_spline"]);
plt =
    data(df) * mapping(:x, :y) * visual(Scatter, markersize = 5, color = "gray") +
    data(dfp) * mapping(:x, :p_linear) * visual(Lines, markersize = 5, color = "navy") +
    data(dfp) * mapping(:x, :p_spline) * visual(Lines, markersize = 5, color = "darkgreen")
draw(plt)
