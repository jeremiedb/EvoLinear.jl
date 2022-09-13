mutable struct EvoLinearRegressor{T<:AbstractFloat,S<:Int}
    nrounds::S
    lambda::T
    rowsample::T
    colsample::T
    nbins::S
    metric::Symbol
    rng
    device
end

function EvoLinearRegressor(;
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

    model = EvoLinearRegressor(nrounds, T(lambda), T(rowsample), T(colsample), nbins, metric, rng, device)

    return model
end

mutable struct EvoLearner
    coef
    bias
end
