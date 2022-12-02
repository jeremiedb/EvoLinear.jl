const loss_fn = Dict(
    MSE => Flux.Losses.mse,
    Logistic => Flux.Losses.logitbinarycrossentropy,
    # :poisson => Poisson,
    # :gamma => Gamma,
    # :tweedie => Tweedie,
)