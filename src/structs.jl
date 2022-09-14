abstract type Loss end
struct MSE <: Loss end
struct Logistic <: Loss end

const loss_types = Dict(
    :mse => MSE,
    :logistic => Logistic
)

mutable struct EvoLinearRegressor{T<:AbstractFloat,I<:Int,S<:Symbol}
    loss::S
    updater::S
    nrounds::I
    L1::T
    L2::T
    nbins::I
    metric::S
    rng
    device::S
end

function EvoLinearRegressor(;
    loss=:mse,
    updater=:all,
    nrounds=10,
    L1=0.0, #
    L2=0.0, #
    nbins=32,
    metric=:mse,
    rng=123,
    device=:cpu)

    T = Float32
    # rng = mk_rng(rng)::Random.AbstractRNG

    model = EvoLinearRegressor(loss, updater, nrounds, T(L1), T(L2), nbins, metric, rng, device)

    return model
end

mutable struct EvoLinearModel{L<:Loss}
    coef
    bias
end
