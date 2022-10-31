using Revise
using EvoLinear
using AlgebraOfGraphics, GLMakie
using DataFrames
using BenchmarkTools
using Distributions

nobs = 1_000
nfeats = 1
T = Float32

x = rand(T, nobs, nfeats) .* 2
coef = randn(T, nfeats)

y = sin.(x[:, 1]) .+ randn(T, nobs) .* 0.1f0
df = DataFrame(hcat(x, y), ["x", "y"]);
draw(data(df) * mapping(:x, :y) * visual(Scatter, markersize = 5, color = "gray"))

config = EvoLinear.EvoLinearRegressor(nrounds = 10, loss = :mse, L1 = 0e-1, L2 = 1)
@time ml = EvoLinear.fit(config; x, y, metric = :mae);

x_pred = reshape(range(start = minimum(x), stop = maximum(x), length = 100), :, 1)
p = EvoLinear.predict_proj(ml, x_pred)

dfp = DataFrame(hcat(x_pred, p), ["x", "p"]);
plt =
    data(df) * mapping(:x, :y) * visual(Scatter, markersize = 5, color = "gray") +
    data(dfp) * mapping(:x, :p) * visual(Lines, markersize = 5, color = "navy")
draw(plt)

using Flux
using Optimisers
using Flux: update!, @functor
using SparseArrays
using LinearAlgebra

# sparse diagonal => project linear effects
sp = sparse(I, 5, 5)
# sparse entries for spline effects
sp = vcat(sp, spzeros(3, 5))
# sparse_proj = Masking sparse matrix allowing to duplicate features to their number of components
#  = default to Diagonal Matrix. Dims = [out_num_dplicated_features, in_num_features]
struct Spline{S,B,W,F}
    mat::S
    b::B
    w::W
    act::F
end
@functor Spline
Flux.trainable(m::Spline) = (b = m.b, w = m.w)
function (m::Spline)(x)
    m.w * m.act.(m.mat * x .- m.b)
end
function Spline(; nfeats, knots = Dict{Int,Int}(), act = relu)
    nknots = sum(values(knots))
    sp = spzeros(nknots, nfeats)
    b = randn(nknots)
    cum = 0
    for (k, v) in knots
        sp[cum+1:cum+v, k] .= 1
        b[cum+1:cum+v] .= quantile(Normal(0,1), collect(1:v) ./ (v+1))
        cum += v
    end
    w = randn(1, nknots) ./ 100 ./ sqrt(nfeats)
    m = Spline(sp, b, w, act)
    return m
end

struct Linear{B,W}
    b::B
    w::W
end
@functor Linear
Flux.trainable(m::Linear) = (b = m.b, w = m.w)
function (m::Linear)(x)
    m.w * x .+ m.b
end
function Linear(; nfeats, mean=0)
    b = ones(1) .* mean
    w = randn(1, nfeats) ./ 100 ./ sqrt(nfeats)
    m = Linear(b, w)
    return m
end

struct SplineModel{L,S,B}
    bn::B
    linear::L
    spline::S
end
@functor SplineModel
Flux.trainable(m::SplineModel) = (linear = m.linear, spline = m.spline)
function (m::SplineModel)(x)
    _x = x |> m.bn
    if isnothing(m.spline)
        m.linear(_x) |> vec
    else
        (m.linear(_x) .+ m.spline(_x)) |> vec
    end
end
function SplineModel(; nfeats, knots = nothing, act = elu, mean=0, affine=false)
    linear = Linear(; nfeats, mean)
    spline = isnothing(knots) ? nothing : Spline(; nfeats, knots, act)
    m = SplineModel(BatchNorm(nfeats; affine), linear, spline)
    return m
end
m = SplineModel(; nfeats = 1, knots = Dict(1 => 4), mean = mean(y), act = elu)
# m = SplineModel(; nfeats = 1, knots = Dict(1 => 8), act = elu)
opts = Optimisers.setup(Optimisers.Adam(1e-1), m)

loss = Flux.Losses.mse
function train!(loss, m, data, opts)
    for (x, y) in data
        grad = gradient(m -> loss(m(x), y), m)[1]
        opts, m = Optimisers.update!(opts, m, grad)  # at every step
    end
end

x1 = rand(1, 100)
@time m(x1);

dtrain = Flux.DataLoader((x = x', y = y), batchsize = 200)

train!(loss, m, dtrain, opts)
p1 = EvoLinear.predict_proj(ml, x_pred)
p2 = m(x_pred')
dfp = DataFrame(hcat(x_pred, p1, p2), ["x", "p_linear", "p_spline"]);
plt =
    data(df) * mapping(:x, :y) * visual(Scatter, markersize = 5, color = "gray") +
    data(dfp) * mapping(:x, :p_linear) * visual(Lines, markersize = 5, color = "navy") + 
    data(dfp) * mapping(:x, :p_spline) * visual(Lines, markersize = 5, color = "darkgreen")
draw(plt)
