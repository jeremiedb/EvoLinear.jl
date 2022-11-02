function init(config::EvoLinearRegressor{T}, x, y; w = nothing) where {T}
    cache = init_cache(config, x, y; w)
    m = EvoLinearModel(config.loss; coef = zeros(T, size(x, 2)), bias = zero(T))
    return m, cache
end

function init_cache(::EvoLinearRegressor{T}, x, y; w = nothing) where {T}
    ∇¹, ∇² = zeros(T, size(x, 2)), zeros(T, size(x, 2))
    ∇b = zeros(T, 2)
    w = isnothing(w) ? ones(T, size(y)) : convert(Vector{T}, w)
    ∑w = sum(w)
    cache = (
        ∇¹ = ∇¹,
        ∇² = ∇²,
        ∇b = ∇b,
        x = convert(Matrix{T}, x),
        y = convert(Vector{T}, y),
        w = w,
        ∑w = ∑w,
        info = Dict(:nrounds => 0),
    )
    return cache
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
    config::EvoLinearRegressor{T};
    x_train,
    y_train,
    w_train = nothing,
    x_eval = nothing,
    y_eval = nothing,
    w_eval = nothing,
    metric = nothing,
    print_every_n = 9999,
    early_stopping_rounds = 9999,
    verbosity = 1,
    fnames = nothing,
    return_logger = false
) where {T}

    m, cache = init(config::EvoLinearRegressor, x_train, y_train; w = w_train)

    logger = nothing
    if !isnothing(metric) && !isnothing(x_eval) && !isnothing(y_eval)
        cb = CallBackLinear(; x_eval, y_eval, w_eval, metric, T)
        logger = init_logger(;
            T,
            metric,
            maximise = is_maximise(cb.feval),
            early_stopping_rounds,
        )
        cb(logger, 0, m)
        (verbosity > 0) && @info "initialization" metric = logger[:metrics][end]
    end

    for iter = 1:config.nrounds
        fit!(m, cache, config)
        if !isnothing(logger)
            cb(logger, iter, m)
            if iter % print_every_n == 0 && verbosity > 0
                @info "iter $iter" metric = logger[:metrics][end]
            end
            (logger[:iter_since_best] >= logger[:early_stopping_rounds]) && break
        end
    end
    if return_logger
        return (m, logger)
    else
        return m
    end
end

function fit!(m::EvoLinearModel{L}, cache, config::EvoLinearRegressor) where {L}

    ∇¹, ∇², ∇b = cache.∇¹ .* 0, cache.∇² .* 0, cache.∇b .* 0
    x, y, w = cache.x, cache.y, cache.w
    ∑w = cache.∑w

    if config.updater == :all
        ####################################################
        # update all coefs then bias
        ####################################################
        p = m(x; proj = true)
        update_∇_bias!(L, ∇b, x, y, p, w)
        update_bias!(m, ∇b)

        p = m(x; proj = true)
        update_∇!(L, ∇¹, ∇², x, y, p, w)
        update_coef!(m, ∇¹, ∇², ∑w, config)
    else
        @error "invalid updater"
    end
    cache[:info][:nrounds] += 1
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
