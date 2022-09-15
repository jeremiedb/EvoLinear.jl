abstract type Loss end
struct MSE <: Loss end
struct Logistic <: Loss end
struct Poisson <: Loss end
struct Gamma <: Loss end
struct Tweedie <: Loss end

const loss_types = Dict(
    :mse => MSE,
    :logistic => Logistic,
    :poisson => Poisson,
    :gamma => Gamma,
    :tweedie => Tweedie
)

mutable struct EvoLinearRegressor{T<:AbstractFloat,I<:Int,S<:Symbol}
    loss::S
    updater::S
    nrounds::I
    L1::T
    L2::T
    metric::S
    rng
    device::S
end

"""
    EvoLinearRegressor(; kwargs...)

- `loss=:mse`: loss function to be minimised. 
    Can be one of:
    
    - `:mse`
    - `:logistic`
    - `:poisson`
    - `:gamma`
    - `:tweedie`

- `nrounds=10`: maximum number of training rounds.
- `L1=0`: Regularization penalty applied by shrinking to 0 weight update if update is < L1. No penalty if update > L1. Results in sparse feature selection. Typically in the `[0, 1]` range on normalized features.
- `L2=0`: Regularization penalty applied to the squared of the weight update value. Restricts large parameter values. Typically in the `[0, 1]` range on normalized features.
- `metric=:mse`: evaluation metric to be tracked. Not used at the moment, use `:metric` in fit[@ref] instead.
- `rng=123`: random seed. Not used at the moment.
- `updater=:all`: training method. Only `:all` is supported at the moment. Gradients for each feature are computed simultaneously, then bias is updated based on all features update. 
- `device=:cpu`: Only `:cpu` is supported at the moment.

"""
function EvoLinearRegressor(;
    loss=:mse,
    updater=:all,
    nrounds=10,
    L1=0.0, #
    L2=0.0, #
    metric=:mse,
    rng=123,
    device=:cpu)

    T = Float32
    # rng = mk_rng(rng)::Random.AbstractRNG

    model = EvoLinearRegressor(loss, updater, nrounds, T(L1), T(L2), metric, rng, device)

    return model
end

mutable struct EvoLinearModel{L<:Loss}
    coef
    bias
end
