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

function Spline(; nfeats, act, knots = Dict{Int,Int}())
    T = Float32
    nknots = sum(values(knots))
    sp = spzeros(T, nknots, nfeats)
    b = randn(T, nknots)
    cum = 0
    for (k, v) in knots
        sp[cum+1:cum+v, k] .= 1
        b[cum+1:cum+v] .= quantile.(Normal(0, 1), collect(1:v) ./ (v + 1))
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

struct EvoSplineModel{L,A,B,C}
    loss::Type{L}
    bn::A
    linear::B
    spline::C
end

@functor EvoSplineModel
Flux.trainable(m::EvoSplineModel) = (bn = m.bn, linear = m.linear, spline = m.spline)

const act_dict = Dict(
    :sigmoid => Flux.sigmoid_fast,
    :tanh => tanh,
    :relu => relu,
    :elu => elu,
    :gelu => gelu,
    :softplus => softplus,
)

function EvoSplineModel(config::EvoSplineRegressor{L,T}; nfeats, mean = 0) where {L,T}
    bn = BatchNorm(nfeats; affine = true)
    linear = Linear(; nfeats, mean)
    act = act_dict[config.act]
    spline = isnothing(config.knots) ? nothing : Spline(; nfeats, config.knots, act)
    m = EvoSplineModel{L,typeof(bn),typeof(linear),typeof(spline)}(L, bn, linear, spline)
    return m
end

get_loss_type(::EvoSplineModel{L,A,B,C}) where {L,A,B,C} = L
get_loss_type(::EvoSplineRegressor{L,T}) where {L,T} = L

function (m::EvoSplineModel{L,A,B,C})(x; proj = true) where {L,A,B,C}
    _x = x |> m.bn
    if isnothing(m.spline)
        p = m.linear(_x) |> vec
    else
        p = (m.linear(_x) .+ m.spline(_x)) |> vec
    end
    proj ? proj!(L, p) : nothing
    return p
end
function (m::EvoSplineModel{L,A,B,C})(p, x; proj = true) where {L,A,B,C}
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
