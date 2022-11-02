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
