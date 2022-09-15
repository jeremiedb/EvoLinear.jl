var documenterSearchIndex = {"docs":
[{"location":"api/#API","page":"API","title":"API","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"EvoLinearRegressor","category":"page"},{"location":"api/#EvoLinear.EvoLinearRegressor","page":"API","title":"EvoLinear.EvoLinearRegressor","text":"EvoLinearRegressor(; kwargs...)\n\nloss=:mse: loss function to be minimised.    Can be one of:\n:mse\n:logistic\n:poisson\n:gamma\n:tweedie\nnrounds=10: maximum number of training rounds.\nL1=0: Regularization penalty applied by shrinking to 0 weight update if update is < L1. No penalty if update > L1. Results in sparse feature selection. Typically in the [0, 1] range on normalized features.\nL2=0: Regularization penalty applied to the squared of the weight update value. Restricts large parameter values. Typically in the [0, 1] range on normalized features.\nmetric=:mse: evaluation metric to be tracked. Not used at the moment, use :metric in fit[@ref] instead.\nrng=123: random seed. Not used at the moment.\nupdater=:all: training method. Only :all is supported at the moment. Gradients for each feature are computed simultaneously, then bias is updated based on all features update. \ndevice=:cpu: Only :cpu is supported at the moment.\n\n\n\n\n\n","category":"type"},{"location":"api/","page":"API","title":"API","text":"EvoLinear.fit\nEvoLinear.predict_linear\nEvoLinear.predict_proj","category":"page"},{"location":"api/#EvoLinear.fit","page":"API","title":"EvoLinear.fit","text":"fit(config::EvoLinearRegressor;\n    x, y, w=nothing,\n    x_eval=nothing, y_eval=nothing, w_eval=nothing,\n    metric=:mse,\n    print_every_n=1,\n    tol=1e-5)\n\nProvided aconfig, EvoLinear.fit takes x and y as features and target inputs, plus optionally w as weights and train a Linear boosted model.\n\nArguments\n\nconfig::EvoLinearRegressor: \n\nKeyword arguments\n\nx::AbstractMatrix: Features matrix. Dimensions are [nobs, num_features].\ny::AbstractVector: Vector of observed targets.\nw=nothing: Vector of weights. Can be be either a Vector or nothing. If nothing, assumes a vector of 1s. \nmetric=:mse: Evaluation metric to be tracked through each iteration.\n\n\n\n\n\n","category":"function"},{"location":"api/#EvoLinear.predict_linear","page":"API","title":"EvoLinear.predict_linear","text":"predict_linear(m, x)\n\nReturns the predictions on the linear basis from model m using the features matrix x.\n\nArguments\n\nm::EvoLinearModel\nx\n\n\n\n\n\n","category":"function"},{"location":"api/#EvoLinear.predict_proj","page":"API","title":"EvoLinear.predict_proj","text":"predict_proj(m, x)\n\nReturns the predictions on the projected basis from model m using the features matrix x.\n\nMSE: predproj = predlinear\nLogistic: predproj = sigmoid(predlinear)\nPoisson: predproj = exp(predlinear)\nGamma: predproj = exp(predlinear)\nTweedie: predproj = exp(predlinear)\n\nArguments\n\nm::EvoLinearModel\nx\n\n\n\n\n\n","category":"function"},{"location":"#EvoLinear.jl","page":"Home","title":"EvoLinear.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"ML library implementing linear boosting with L1 and L2 regularization.","category":"page"},{"location":"","page":"Home","title":"Home","text":"For Tree based boosting, consider EvoTrees.jl.","category":"page"},{"location":"","page":"Home","title":"Home","text":"Supported loss functions:","category":"page"},{"location":"","page":"Home","title":"Home","text":"mse (squared-error)\nlogistic (logloss) regression\npoisson\ngamma\ntweedie","category":"page"},{"location":"#Installation","page":"Home","title":"Installation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"pkg> add https://github.com/jeremiedb/EvoLinear.jl","category":"page"},{"location":"#Getting-started","page":"Home","title":"Getting started","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Build a configuration struct with EvoLinearRegressor. Then EvoLinear.fit takes x::Matrix and y::Vector as inputs, plus optionally w::Vector as weights and fits a linear boosted model.","category":"page"},{"location":"","page":"Home","title":"Home","text":"using EvoLinear\nconfig = EvoLinearRegressor(loss=:mse, L1=1e-1, L2=1e-2, nrounds=10)\nm = EvoLinear.fit(config; x, y, metric=:mse)\np = EvoLinear.predict_proj(m, x)","category":"page"}]
}
