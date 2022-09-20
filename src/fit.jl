function init(config::EvoLinearRegressor{T};
    x, y, w=nothing) where {T}

    cache = init_cache(config; x, y, w)
    m = EvoLinearModel(config.loss; coef=zeros(T, size(x, 2)), bias=zero(T))

    return m, cache

end


function init_cache(::EvoLinearRegressor{T};
    x, y, w=nothing) where {T}

    ∇¹, ∇² = zeros(T, size(x, 2)), zeros(T, size(x, 2))
    ∇b = zeros(T, 2)

    isnothing(w) ? w = ones(T, size(x, 1)) : convert(Vector{T}, w)
    ∑w = sum(w)

    cache = (
        ∇¹=∇¹, ∇²=∇², ∇b=∇b,
        # x=x, y=y, w=w,
        x=convert(Matrix{T}, x), y=convert(Vector{T}, y), w=w,
        ∑w=∑w,
        logger=Dict(:nrounds => 0)
    )
    return cache

end


"""
    fit(config::EvoLinearRegressor;
        x, y, w=nothing,
        x_eval=nothing, y_eval=nothing, w_eval=nothing,
        metric=:none,
        print_every_n=1,
        tol=1e-5)

Provided a `config`, `EvoLinear.fit` takes `x` and `y` as features and target inputs, plus optionally `w` as weights and train a Linear boosted model.

# Arguments
- `config::EvoLinearRegressor`: 

# Keyword arguments
- `x::AbstractMatrix`: Features matrix. Dimensions are `[nobs, num_features]`.
- `y::AbstractVector`: Vector of observed targets.
- `w=nothing`: Vector of weights. Can be be either a `Vector` or `nothing`. If `nothing`, assumes a vector of 1s. 
- `metric=nothing`: Evaluation metric to be tracked through each iteration. Can be one of:

    - `:mse`
    - `:logistic`
    - `:poisson_deviance`
    - `:gamma_deviance`
    - `:tweedie_deviance`
"""
function fit(config::EvoLinearRegressor;
    x, y, w=nothing,
    x_eval=nothing, y_eval=nothing, w_eval=nothing,
    metric=nothing,
    print_every_n=1,
    tol=1e-5)

    m, cache = init(config::EvoLinearRegressor; x, y, w)

    if !isnothing(metric)
        metric_f = metric_dict[metric]
        p = m(x, proj=true)
        tracker = metric_f(p, y)
        @info "initial $metric:" tracker
    end

    for i in 1:config.nrounds
        fit!(m, cache, config)

        if !isnothing(metric) && i % print_every_n == 0
            p = m(x; proj=true)
            tracker = metric_f(p, y)
            @info "$metric iter $i:" tracker
        end
    end

    return m
end

function fit!(m::EvoLinearModel{L}, cache, config::EvoLinearRegressor) where {L}

    ∇¹, ∇², ∇b = cache.∇¹ .* 0, cache.∇² .* 0, cache.∇b .* 0
    x, y, w = cache.x, cache.y, cache.w
    ∑w = cache.∑w

    if config.updater == :all
        ####################################################
        # update all coefs then bias
        ####################################################
        p = m(x; proj=true)
        update_∇_bias!(L, ∇b, x, y, p, w)
        update_bias!(m, ∇b)

        p = m(x; proj=true)
        update_∇!(L, ∇¹, ∇², x, y, p, w)
        update_coef!(m, ∇¹, ∇², ∑w, config)

    elseif config.updater == :single
        @error "single update needs to be fixed - preds update needs linear projection basis"
        ####################################################
        # update bias following each feature update
        ####################################################
        # p = predict_proj(m, x)
        # for feat in axes(x, 2)
        #     update_∇!(L, ∇¹, ∇², x, y, p, w, feat)
        #     Δ_coef = coef_update(m, ∇¹, ∇², feat)
        #     p .+= Δ_coef .* x[:, feat]
        #     m.coef[feat] += Δ_coef
        # end
        # update_∇_bias!(L, ∇b, x, y, p, w)
        # update_bias!(m, ∇b)
    else
        @error "invalid updater"
    end
    cache[:logger][:nrounds] += 1
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
