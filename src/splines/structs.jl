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
    T = Float32
    nknots = sum(values(knots))
    sp = spzeros(T, nknots, nfeats)
    b = randn(T, nknots)
    cum = 0
    for (k, v) in knots
        sp[cum+1:cum+v, k] .= 1
        b[cum+1:cum+v] .= quantile(Normal(0, 1), collect(1:v) ./ (v + 1))
        cum += v
    end
    w = randn(T, 1, nknots) ./ T(100) ./ T(sqrt(nfeats))
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

function Linear(; nfeats, mean = 0)
    T = Float32
    b = ones(T, 1) .* T(mean)
    w = randn(T, 1, nfeats) ./ T(100) ./ T(sqrt(nfeats))
    m = Linear(b, w)
    return m
end

struct SplineModel{L,A,B,C}
    loss::Type{L}
    bn::A
    linear::B
    spline::C
end

@functor SplineModel
Flux.trainable(m::SplineModel) = (linear = m.linear, spline = m.spline)

function SplineModel(
    ::EvoSplineRegressor{L,T};
    nfeats,
    knots = nothing,
    act = elu,
    mean = 0,
    affine = true,
) where {L,T}
    bn = BatchNorm(nfeats; affine)
    linear = Linear(; nfeats, mean)
    spline = isnothing(knots) ? nothing : Spline(; nfeats, knots, act)
    m = SplineModel{L, typeof(bn), typeof(linear), typeof(spline)}(L, bn, linear, spline)
    return m
end

function (m::SplineModel{L,A,B,C})(x; proj=true) where {L,A,B,C}
    _x = x |> m.bn
    if isnothing(m.spline)
        p = m.linear(_x) |> vec
    else
        p = (m.linear(_x) .+ m.spline(_x)) |> vec
    end
    proj ? proj!(L, p) : nothing
    return p
end
function (m::SplineModel{L,A,B,C})(p, x; proj=true) where {L,A,B,C}
    _x = x |> m.bn
    if isnothing(m.spline)
        p .= m.linear(_x) |> vec
    else
        p .= (m.linear(_x) .+ m.spline(_x)) |> vec
    end
    proj ? proj!(L, p) : nothing
    return nothing
end

function proj!(::L, p) where {L<:Type{MSE}}
    return nothing
end
function proj!(::L, p) where {L<:Type{Logistic}}
    p .= sigmoid.(p)
    return nothing
end
function proj!(::L, p) where {L<:Union{Type{Poisson},Type{Gamma},Type{Tweedie}}}
    p .= exp.(p)
    return nothing
end
