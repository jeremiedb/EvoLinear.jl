function init(config::EvoLinearRegressor{T}; x, y, w = nothing) where {T}
    cache = init_cache(config; x, y, w)
    m = EvoLinearModel(config.loss; coef = zeros(T, size(x, 2)), bias = zero(T))
    return m, cache
end

function init_cache(::EvoLinearRegressor{T}; x, y, w = nothing) where {T}
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
    config::EvoLinearRegressor{T,I,S};
    x,
    y,
    w = nothing,
    x_eval = nothing,
    y_eval = nothing,
    w_eval = nothing,
    metric_name = nothing,
    print_every_n = 9999,
    early_stopping_rounds = 9999,
    fnames = nothing,
) where {T,I,S}

    m, cache = init(config::EvoLinearRegressor; x, y, w)

    logger = nothing
    if !isnothing(metric_name) && !isnothing(x_eval) && !isnothing(y_eval)
        cb = CallBackLinear(; x_eval, y_eval, w_eval, metric_name, T)
        logger = init_logger(;
            T,
            metric_name,
            maximise = is_maximise(cb.feval),
            early_stopping_rounds,
        )
        cb(logger, 0, m)
        @info "initial metric" metric = logger[:metrics][end]
    end

    for i = 1:config.nrounds
        fit!(m, cache, config)
        if !isnothing(logger)
            cb(logger, i, m)
            if i % print_every_n == 0
                @info "iter $i" metric = logger[:metrics][end]
            end
        end
    end

    return m, logger
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


struct CallBackLinear{F,M,V}
    feval::F
    x::M
    p::V
    y::V
    w::V
end
function (cb::CallBackLinear)(logger, iter, m)
    m(cb.p, cb.x; proj = true)
    metric = cb.feval(cb.p, cb.y, cb.w)
    update_logger!(logger, iter, metric)
    return logger[:early_stopping_rounds] <= logger[:iter_since_best]
end

function CallBackLinear(; metric_name, x_eval, y_eval, w_eval = nothing, T = Float32)
    feval = metric_dict[metric_name]
    x = convert(Matrix{T}, x_eval)
    p = zeros(T, length(y_eval))
    y = convert(Vector{T}, y_eval)
    w = isnothing(w_eval) ? ones(T, size(y)) : convert(Vector{T}, w_eval)
    return CallBackLinear(feval, x, p, y, w)
end

function init_logger(; T, metric_name, maximise, early_stopping_rounds)
    logger = Dict(
        :name => String(metric_name),
        :maximise => maximise,
        :early_stopping_rounds => early_stopping_rounds,
        :niter => 0,
        :iter => Int[],
        :metrics => T[],
        :iter_since_best => 0,
        :best_iter => 0,
        :best_metric => 0.0,
    )
    return logger
end

function update_logger!(logger, iter, metric)
    logger[:niter] = iter
    push!(logger[:iter], iter)
    push!(logger[:metrics], metric)
    if iter > 0
        if (logger[:maximise] && metric > logger[:best_metric]) ||
           (!logger[:maximise] && metric < logger[:best_metric])
            logger[:best_metric] = metric
            logger[:best_iter] = iter
            logger[:iter_since_best] = 0
        else
            logger[:iter_since_best] += logger[:iter][end] - logger[:iter][end-1]
        end
    end
end
