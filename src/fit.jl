function init(config::EvoLinearRegressor;
    x, y, w=nothing)

    T = Float32
    loss = loss_types[config.loss]
    m = EvoLinearModel{loss}(zeros(T, size(x, 2)), zero(T))
    p_linear = predict_linear(m, x)
    p_proj = predict_proj(m, x)

    ∇¹ = init_∇¹(x)
    ∇² = init_∇²(x)
    ∇b = zeros(T, 2)

    isnothing(w) ? w = ones(T, size(x, 1)) : nothing
    ∑w = sum(w)

    cache = (
        p_linear=p_linear, p_proj=p_proj,
        ∇¹=∇¹, ∇²=∇², ∇b=∇b,
        x=x, y=y, w=w,
        ∑w=∑w
    )
    return m, cache

end


"""
    fit(config::EvoLinearRegressor;
        x, y, w=nothing,
        x_eval=nothing, y_eval=nothing, w_eval=nothing,
        metric=:mse,
        print_every_n=1,
        tol=1e-5)

`Provided a `config`, EvoLinear.fit` takes `x` and `y` as features and target inputs, plus optionally `w` as weights and train a Linear boosted model.

# Arguments
- `config::EvoLinearRegressor`: 

# Keyword arguments
- `x::AbstractMatrix`: Features matrix. Dimensions are `[nobs, num_features]`.
- `y::AbstractVector`: Vector of observed targets.
- `w=nothing`: Vector of weights. Can be be either a `Vector` or `nothing`. If `nothing`, assumes a vector of 1s. 
- `metric=:mse`: Evaluation metric to be tracked through each iteration.
"""
function fit(config::EvoLinearRegressor;
    x, y, w=nothing,
    x_eval=nothing, y_eval=nothing, w_eval=nothing,
    metric=:mse,
    print_every_n=1,
    tol=1e-5)

    m, cache = init(config::EvoLinearRegressor; x, y, w)

    metric_f = metric_dict[metric]
    p = predict_proj(m, x)
    tracker = metric_f(p, y)
    @info "initial $metric:" tracker

    for i in 1:config.nrounds
        fit!(m, cache, config)

        if i % print_every_n == 0
            p = predict_proj(m, x)
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
        p = predict_proj(m, x)
        update_∇!(L, ∇¹, ∇², x, y, p, w)
        update_coef!(m, ∇¹, ∇², ∑w, config.L1, config.L2)
        p = predict_proj(m, x)
        update_∇_bias!(L, ∇b, x, y, p, w)
        update_bias!(m, ∇b)
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
    return nothing
end

function update_coef!(m, ∇¹, ∇², ∑w, L1, L2)
    update = -∇¹ ./ (∇² .+ L2 * ∑w)
    update[abs.(update).<L1] .= 0
    m.coef .+= update
    return nothing
end
function update_bias!(m, ∇b)
    m.bias += -∇b[1] / ∇b[2]
    return nothing
end

