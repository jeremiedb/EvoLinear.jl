function init(config::EvoSplineRegressor{L,T}, x, y; w = nothing) where {L,T}

    @info "starting spline"
    device = config.device == :cpu ? Flux.cpu : Flux.gpu
    nfeats = size(x, 2)
    dtrain = DataLoader(
        (x = Matrix{T}(x') |> device, y = T.(y) |> device),
        batchsize = config.batchsize,
    )
    loss = loss_fn[L]

    m = EvoSplineModel(config; nfeats, mean = mean(y)) |> device

    opt = Optimisers.NAdam(config.eta)
    opts = Optimisers.setup(opt, m)

    cache = (dtrain = dtrain, loss = loss, opts = opts, info = Dict(:nrounds => 0))
    return m, cache
end


"""
    fit(config::EvoSplineRegressor; x_train, y_train, x_eval = nothing, y_eval = nothing)

Train a splined linear model. 
"""
function fit(
    config::EvoSplineRegressor{L,T};
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
    return_logger = false,
) where {L,T}

    m, cache = init(config, x_train, y_train; w = w_train)

    logger = nothing
    if !isnothing(metric) && !isnothing(x_eval) && !isnothing(y_eval)
        cb = CallBackLinear(config; metric, x_eval, y_eval, w_eval)
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
        fit!(m, cache)
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

function fit!(m, cache)
    for d in cache[:dtrain]
        grads = gradient(model -> cache[:loss](model(d[:x]; proj = false), d[:y]), m)[1]
        Optimisers.update!(cache[:opts], m, grads)
    end
    cache[:info][:nrounds] += 1
    return nothing
end

function CallBackLinear(
    config::EvoSplineRegressor{L,T};
    metric,
    x_eval,
    y_eval,
    w_eval = nothing,
) where {L,T}
    device = config.device == :cpu ? Flux.cpu : Flux.gpu
    feval = metric_dict[metric]
    x = convert(Matrix{T}, x_eval')
    p = zeros(T, length(y_eval))
    y = convert(Vector{T}, y_eval)
    w = isnothing(w_eval) ? ones(T, size(y)) : convert(Vector{T}, w_eval)
    return CallBackLinear(feval, x |> device, p |> device, y |> device, w |> device)
end
