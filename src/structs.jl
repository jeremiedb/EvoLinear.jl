abstract type Loss end
struct MSE <: Loss end
struct Logistic <: Loss end

const loss_types = Dict(
    :mse => MSE,
    :logistic => Logistic
)

mutable struct EvoLinearRegressor{T<:AbstractFloat,I<:Int}
    loss::Symbol
    nrounds::I
    lambda::T
    rowsample::T
    colsample::T
    nbins::I
    metric::Symbol
    rng
    device
end

function EvoLinearRegressor(;
    loss=:mse,
    nrounds=10,
    lambda=0.0, #
    rowsample=1.0,
    colsample=1.0,
    nbins=32,
    metric=:mse,
    rng=123,
    device="cpu")

    T = Float32
    # rng = mk_rng(rng)::Random.AbstractRNG

    model = EvoLinearRegressor(loss, nrounds, T(lambda), T(rowsample), T(colsample), nbins, metric, rng, device)

    return model
end

mutable struct EvoLinearModel{L<:Loss}
    coef
    bias
end
