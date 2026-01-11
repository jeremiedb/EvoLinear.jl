mutable struct EvoLinearRegressor <: MMI.Deterministic
    loss::Symbol
    metric::Symbol
    updater::Symbol
    nrounds::Int
    eta::Float32
    L1::Float32
    L2::Float32
    early_stopping_rounds::Int
    seed::Int
end

"""
    EvoLinearRegressor(; kwargs...)


A model type for constructing a EvoLinearRegressor, based on [EvoLinear.jl](https://github.com/jeremiedb/EvoLinear.jl), and implementing both an internal API and the MLJ model interface.

# Keyword arguments

- `loss=:mse`: loss function to be minimised. 
    Can be one of:

    - `:mse`
    - `:logloss`
    - `:poisson`
    - `:gamma`
    - `:tweedie`

- `nrounds=10`: maximum number of training rounds.
- `eta=1`: Learning rate. Typically in the range `[1e-2, 1]`.
- `L1=0`: Regularization penalty applied by shrinking to 0 weight update if update is < L1. No penalty if update > L1. Results in sparse feature selection. Typically in the `[0, 1]` range on normalized features.
- `L2=0`: Regularization penalty applied to the squared of the weight update value. Restricts large parameter values. Typically in the `[0, 1]` range on normalized features.
- `seed::Int=123`: random seed.
- `updater=:all`: training method. Only `:all` is supported at the moment. Gradients for each feature are computed simultaneously, then bias is updated based on all features update. 
- `device=:cpu`: Only `:cpu` is supported at the moment.

# Internal API

Use `config = EvoLinearRegressor()` to construct an hyper-parameter struct with default hyper-parameters.
Provide keyword arguments as listed above to override defaults, for example:

```julia
EvoLinearRegressor(loss=:logloss, L1=1e-3, L2=1e-2, nrounds=100)
```

## Training model

A model is built using [`fit`](@ref):

```julia
config = EvoLinearRegressor()
m = fit(config; x, y, w)
```

## Inference

Fitted results is an `EvoLinearModel` which acts as a prediction function when passed a features matrix as argument.  

```julia
preds = m(x)
```

# MLJ Interface

From MLJ, the type can be imported using:

```julia
EvoLinearRegressor = @load EvoLinearRegressor pkg=EvoLinear
```

Do `model = EvoLinearRegressor()` to construct an instance with default hyper-parameters.
Provide keyword arguments to override hyper-parameter defaults, as in `EvoLinearRegressor(loss=...)`.

## Training model

In MLJ or MLJBase, bind an instance `model` to data with `mach = machine(model, X, y)` where: 

- `X`: any table of input features (eg, a `DataFrame`) whose columns
  each have one of the following element scitypes: `Continuous`,
  `Count`, or `<:OrderedFactor`; check column scitypes with `schema(X)`
- `y`: is the target, which can be any `AbstractVector` whose element
  scitype is `<:Continuous`; check the scitype
  with `scitype(y)`

Train the machine using `fit!(mach, rows=...)`.

## Operations

  - `predict(mach, Xnew)`: return predictions of the target given
  features `Xnew` having the same scitype as `X` above. Predictions
  are deterministic.

## Fitted parameters

The fields of `fitted_params(mach)` are:

- `:fitresult`: the `EvoLinearModel` object returned by EvoLnear.jl fitting algorithm.

## Report

The fields of `report(mach)` are:

- `:coef`: Vector of coefficients (βs) associated to each of the features.
- `:bias`: Value of the bias.
- `:names`: Names of each of the features.

"""
function EvoLinearRegressor(; kwargs...)

    # defaults arguments
    args = Dict{Symbol,Any}(
        :loss => :mse,
        :metric => nothing,
        :updater => :all,
        :nrounds => 10,
        :eta => 1,
        :L1 => 0,
        :L2 => 0,
        :early_stopping_rounds => typemax(Int),
        :seed => 123
    )

    args_ignored = setdiff(keys(kwargs), keys(args))
    length(args_ignored) > 0 &&
        @info "The following kwargs are not supported and will be ignored: $(args_ignored)."

    args_override = intersect(keys(args), keys(kwargs))
    for arg in args_override
        args[arg] = kwargs[arg]
    end

    _loss_list = [:mse, :logloss, :poisson, :gamma, :tweedie]
    loss = Symbol(args[:loss])
    if loss ∉ _loss_list
        error("Invalid loss. Must be one of: $_loss_list")
    end

    _metric_list = [:mse, :rmse, :mae, :logloss, :poisson, :gamma, :tweedie]
    if isnothing(args[:metric])
        metric = loss
    else
        metric = Symbol(args[:metric])
    end
    if metric ∉ _metric_list
        error("Invalid metric. Must be one of: $_metric_list")
    end

    model = EvoLinearRegressor(
        loss,
        metric,
        args[:updater],
        args[:nrounds],
        args[:eta],
        args[:L1],
        args[:L2],
        args[:early_stopping_rounds],
        args[:seed])

    return model
end

mutable struct EvoLinearModel{L<:Loss,A,B}
    loss::Type{L}
    coef::A
    bias::B
    info::Dict{Symbol,Any}
end
EvoLinearModel(loss::Type{<:Loss}; coef, bias, info) = EvoLinearModel(loss, coef, bias, info)
EvoLinearModel(loss::Symbol; coef, bias, info) = EvoLinearModel(loss_types[loss]; coef, bias, info)

const EvoLinearTypes = Union{EvoLinearRegressor}
