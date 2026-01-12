function init(learner::EvoLinearRegressor; features, target, weight=nothing)

    T = Float32
    Tables.istable(features) || error("EvoLinear.fit(...) only accepts Tables compatible input for `features` (ex: named tuples, DataFrames...)")
    schema = Tables.schema(features)
    feature_names = collect(schema.names)

    nobs = Tables.DataAPI.nrow(features)
    nfeats = length(feature_names)

    x = zeros(T, nobs, nfeats)
    @threads for j in axes(x, 2)
        @views x[:, j] .= Tables.getcolumn(features, feature_names[j])
    end

    y = convert(Vector{T}, target)
    w = isnothing(weight) ? ones(T, nobs) : V{T}(weight)
    ∑w = sum(w)

    ∇¹, ∇² = zeros(T, size(x, 2)), zeros(T, size(x, 2))
    ∇b = zeros(T, 2)

    info = Dict(
        :nrounds => 0,
        :feature_names => feature_names,
        :target_name => nothing,
        :weight_name => nothing)

    cache = (
        x=x,
        y=y,
        w=w,
        ∑w=∑w,
        ∇¹=∇¹,
        ∇²=∇²,
        ∇b=∇b,
    )

    m = EvoLinearModel(learner.loss; coef=zeros(T, size(x, 2)), bias=zero(T), info=info)

    return m, cache
end


function init(learner::EvoLinearRegressor, dtrain; target_name, feature_names=nothing, weight_name=nothing)

    Tables.istable(dtrain) || error("EvoLinear.fit(...) only accepts Tables compatible input for `dtrain` (ex: named tuples, DataFrames...)")
    schema = Tables.schema(dtrain)
    target_name = Symbol(target_name)
    weight_name = isnothing(weight_name) ? nothing : Symbol(weight_name)

    if isnothing(feature_names)
        feature_names = Symbol[]
        for i in eachindex(schema.names)
            if (schema.types[i] <: Real) && (schema.names[i] ∉ [target_name, weight_name])
                push!(feature_names, schema.names[i])
            end
        end
    else
        isa(feature_names, AbstractVector) || error("feature_names must be a vector")
        feature_names = Symbol.(feature_names)
    end
    @assert isa(feature_names, Vector{Symbol})
    @assert Set(feature_names) <= Set(schema.names)

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

    m = EvoLinearModel(learner.loss; coef=zeros(T, size(x, 2)), bias=zero(T), info=info)

    return m, cache
end


"""
    function fit(
        learner::EvoLinearRegressor,
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

Provided a `config`, `EvoLinear.fit` takes `x` and `y` as features and target inputs, plus optionally `w` as weights and train a Linear boosted model.

# Arguments
- `learner::EvoLinearRegressor`: 
- `dtrain`: A `Tables.jl` compatible table containing the feature, target and optionally weight variables.

# Keyword arguments

- `target_name: 
- `feature_names=nothing: 
- `weight_name=nothing: 
- `deval=nothing: 
- `print_every_n=9999: 
- `verbosity=1: 
"""
function fit(
    learner::EvoLinearRegressor,
    dtrain;
    target_name,
    feature_names=nothing,
    weight_name=nothing,
    deval=nothing,
    print_every_n=9999,
    verbosity=1
)

    m, cache = init(learner, dtrain; feature_names, target_name, weight_name)

    if !isnothing(deval)
        cb = CallBack(learner, deval; feature_names=m.info[:feature_names], target_name, weight_name)
        logger = init_logger(;
            learner.metric,
            maximise=is_maximise(cb.feval),
            learner.early_stopping_rounds,
        )
        update_logger!(logger, m, cb, 0)
        (verbosity > 0) && @info "initialization" metric = logger[:metrics][end]
    else
        logger = nothing
    end

    for iter = 1:learner.nrounds
        fit!(m, cache, learner)
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

function fit!(m::EvoLinearModel, cache, learner::EvoLinearRegressor)

    ∇¹, ∇², ∇b = cache.∇¹ .* 0, cache.∇² .* 0, cache.∇b .* 0
    x, y, w = cache.x, cache.y, cache.w
    ∑w = cache.∑w

    if learner.updater == :all
        # update all coefs then bias
        p = m(x; proj=true)
        update_∇_bias!(m.loss, ∇b, x, y, p, w)
        update_bias!(m, ∇b)

        p = m(x; proj=true)
        update_∇!(m.loss, ∇¹, ∇², x, y, p, w)
        update_coef!(m, ∇¹, ∇², ∑w, learner)
    else
        @error "invalid updater"
    end
    m.info[:nrounds] += 1
    return nothing
end

function update_coef!(m, ∇¹, ∇², ∑w, learner)
    update = -∇¹ ./ (∇² .+ learner.L2 * ∑w)
    update[abs.(update).<learner.L1] .= 0
    m.coef .+= update .* learner.eta
    return nothing
end
function update_bias!(m, ∇b)
    m.bias += -∇b[1] / ∇b[2]
    return nothing
end
