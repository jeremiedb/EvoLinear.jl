mutable struct EvoSplineRegressor{L,T} <: MMI.Deterministic
    nrounds::Int
    opt::Symbol
    batchsize::Int
    act::Symbol
    eta::T
    L2::T
    knots::Union{Dict,Nothing}
    rng::Any
    device::Symbol
end


"""
    EvoSplineRegressor(; kwargs...)


A model type for constructing a EvoLinearRegressor, based on [EvoLinear.jl](https://github.com/jeremiedb/EvoLinear.jl), and implementing both an internal API and the MLJ model interface.

# Keyword arguments

- `loss=:mse`: loss function to be minimised. 
    Can be one of:
    
    - `:mse`
    - `:logistic`
    - `:poisson`
    - `:gamma`
    - `:tweedie`

- `nrounds=10`: maximum number of training rounds.
- `eta=1`: Learning rate. Typically in the range `[1e-2, 1]`.
- `L1=0`: Regularization penalty applied by shrinking to 0 weight update if update is < L1. No penalty if update > L1. Results in sparse feature selection. Typically in the `[0, 1]` range on normalized features.
- `L2=0`: Regularization penalty applied to the squared of the weight update value. Restricts large parameter values. Typically in the `[0, 1]` range on normalized features.
- `rng=123`: random seed. Not used at the moment.
- `updater=:all`: training method. Only `:all` is supported at the moment. Gradients for each feature are computed simultaneously, then bias is updated based on all features update. 
- `device=:cpu`: Only `:cpu` is supported at the moment.

# Internal API

Do `config = EvoLinearRegressor()` to construct an hyper-parameter struct with default hyper-parameters.
Provide keyword arguments as listed above to override defaults, for example:

```julia
EvoLinearRegressor(loss=:logistic, L1=1e-3, L2=1e-2, nrounds=100)
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
function EvoSplineRegressor(; kwargs...)

    # defaults arguments
    args = Dict{Symbol,Any}(
        :loss => :mse,
        :nrounds => 10,
        :opt => :Adam,
        :batchsize => 1024,
        :act => :relu,
        :eta => 1e-3,
        :L2 => 0.0,
        :knots => nothing,
        :rng => 123,
        :device => :cpu,
        :T => Float32,
    )

    args_ignored = setdiff(keys(kwargs), keys(args))
    args_ignored_str = join(args_ignored, ", ")
    length(args_ignored) > 0 &&
        @info "Following $(length(args_ignored)) provided arguments will be ignored: $(args_ignored_str)."

    args_default = setdiff(keys(args), keys(kwargs))
    args_default_str = join(args_default, ", ")
    length(args_default) > 0 &&
        @info "Following $(length(args_default)) arguments were not provided and will be set to default: $(args_default_str)."

    args_override = intersect(keys(args), keys(kwargs))
    for arg in args_override
        args[arg] = kwargs[arg]
    end

    args[:rng] = mk_rng(args[:rng])
    T = args[:T]
    L = loss_types[Symbol(args[:loss])]

    model = EvoSplineRegressor{L,T}(
        args[:nrounds],
        Symbol(args[:opt]),
        args[:batchsize],
        Symbol(args[:act]),
        args[:T](args[:eta]),
        args[:T](args[:L2]),
        args[:knots],
        args[:rng],
        Symbol(args[:device]),
    )

    return model
end
