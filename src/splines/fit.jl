
"""
    fit(config::EvoSplineRegressor; x_train, y_train, x_eval = nothing, y_eval = nothing)

Train a splined linear model. 
"""
function fit(
    config::EvoSplineRegressor;
    x_train,
    y_train,
    w_train = nothing,
    x_eval = nothing,
    y_eval = nothing,
    w_eval = nothing,
)

    nfeats = size(x_train, 2)
    dtrain = DataLoader((x = x_train', y = y_train), batchsize = config.batchsize)
    m = SplineModel(;
        nfeats = nfeats,
        knots = config.knots,
        mean = mean(y_train),
        act = act_dict[config.act],
    )

    opt = Optimisers.Adam(config.eta)
    opts = Optimisers.setup(opt, m)
    loss = Flux.Losses.mse

    for iter = 1:config.nrounds
        fit!(loss, m, dtrain, opts)
    end

    return m
end

function fit!(loss, m, data, opts)
    for d in data
        grad = gradient(m -> loss(m(d[:x]), d[:y]), m)[1]
        opts, m = Optimisers.update!(opts, m, grad)
    end
end

const act_dict = Dict(
    :sigmoid => sigmoid,
    :tanh => tanh,
    :relu => relu,
    :elu => elu,
    :gelu => gelu,
    :softplus => softplus,
)

