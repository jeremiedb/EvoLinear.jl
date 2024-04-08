function init(config::EvoLinearRegressor{L}, dtrain; feature_names, target_name, weight_name=nothing) where {L}

    T = Float32
    nobs = Tables.DataAPI.nrow(dtrain)
    nfeats = length(feature_names)

    x = zeros(T, nobs, nfeats)
    @threads for j in axes(x, 2)
        @views x[:, j] .= Tables.getcolumn(dtrain, feature_names[j])
    end

    y = Tables.getcolumn(dtrain, target_name)
    y = convert(Vector{T}, y)

    w = isnothing(weight_name) ? ones(T, nobs) : convert(Vector{T}, Tables.getcolumn(dtrain, weight_name))
    ∑w = sum(w)

    ∇¹, ∇² = zeros(T, size(x, 2)), zeros(T, size(x, 2))
    ∇b = zeros(T, 2)

    info = Dict(
        :nrounds => 0,
        :feature_names => feature_names,
        :target_name => target_name,
        :weight_name => weight_name)

    cache = (
        x=x,
        y=y,
        w=w,
        ∑w=∑w,
        ∇¹=∇¹,
        ∇²=∇²,
        ∇b=∇b,
    )

    m = EvoLinearModel(L; coef=zeros(T, size(x, 2)), bias=zero(T), info=info)

    return m, cache
end


"""
    fit(config::EvoLinearRegressor;
        x, y, w=nothing,
        x_eval=nothing, y_eval=nothing, w_eval=nothing,
        metric=:none,
        print_every_n=1)

Provided a `config`, `EvoLinear.fit` takes `x` and `y` as features and target inputs, plus optionally `w` as weights and train a Linear boosted model.

# Arguments
- `config::EvoLinearRegressor`: 

# Keyword arguments
- `x::AbstractMatrix`: Features matrix. Dimensions are `[nobs, num_features]`.
- `y::AbstractVector`: Vector of observed targets.
- `w=nothing`: Vector of weights. Can be be either a `Vector` or `nothing`. If `nothing`, assumes a vector of 1s. 
- `metric=nothing`: Evaluation metric to be tracked through each iteration. Default to `nothing`. Can be one of:

    - `:mse`
    - `:logistic`
    - `:poisson_deviance`
    - `:gamma_deviance`
    - `:tweedie_deviance`
"""
function fit(
    config::EvoLinearRegressor,
    dtrain;
    feature_names,
    target_name,
    weight_name=nothing,
    deval=nothing,
    metric=nothing,
    print_every_n=9999,
    early_stopping_rounds=9999,
    verbosity=1
)

    feature_names = Symbol.(feature_names)
    target_name = Symbol(target_name)
    weight_name = isnothing(weight_name) ? nothing : Symbol(weight_name)
    logger = nothing

    m, cache = init(config, dtrain; feature_names, target_name, weight_name)

    if !isnothing(metric) && !isnothing(deval)
        cb = CallBack(config, deval; metric, feature_names, target_name, weight_name)
        logger = init_logger(;
            metric,
            maximise=is_maximise(cb.feval),
            early_stopping_rounds,
        )
        update_logger!(logger, m, cb, 0)
        (verbosity > 0) && @info "initialization" metric = logger[:metrics][end]
    end

    for iter = 1:config.nrounds
        fit!(m, cache, config)
        if !isnothing(logger)
            update_logger!(logger, m, cb, iter)
            if iter % print_every_n == 0 && verbosity > 0
                @info "iter $iter" metric = logger[:metrics][end]
            end
            (logger[:iter_since_best] >= logger[:early_stopping_rounds]) && break
        end
    end

    m.info[:logger] = logger
    return m
end

function fit!(m::EvoLinearModel, cache, config::EvoLinearRegressor)

    ∇¹, ∇², ∇b = cache.∇¹ .* 0, cache.∇² .* 0, cache.∇b .* 0
    x, y, w = cache.x, cache.y, cache.w
    ∑w = cache.∑w

    if config.updater == :all
        # update all coefs then bias
        p = m(x; proj=true)
        update_∇_bias!(m.loss, ∇b, x, y, p, w)
        update_bias!(m, ∇b)

        p = m(x; proj=true)
        update_∇!(m.loss, ∇¹, ∇², x, y, p, w)
        update_coef!(m, ∇¹, ∇², ∑w, config)
    else
        @error "invalid updater"
    end
    m.info[:nrounds] += 1
    return nothing
end

function update_coef!(m, ∇¹, ∇², ∑w, config)
    update = -∇¹ ./ (∇² .+ config.L2 * ∑w)
    update[abs.(update).<config.L1] .= 0
    m.coef .+= update .* config.eta
    return nothing
end
function update_bias!(m, ∇b)
    m.bias += -∇b[1] / ∇b[2]
    return nothing
end

function CallBack(
    config::EvoLinearRegressor,
    deval;
    metric,
    feature_names,
    target_name,
    weight_name=nothing
)
    T = Float32
    nobs = Tables.DataAPI.nrow(deval)
    nfeats = length(feature_names)
    feval = metric_dict[metric]

    x = zeros(T, nobs, nfeats)
    @threads for j in axes(x, 2)
        @views x[:, j] .= Tables.getcolumn(deval, feature_names[j])
    end
    y = Tables.getcolumn(deval, target_name)
    y = convert(Vector{T}, y)
    p = zero(y)

    w = isnothing(weight_name) ? ones(T, nobs) : convert(Vector{T}, Tables.getcolumn(deval, weight_name))

    return CallBack(feval, x, p, y, w)
end
