```@raw html
---
layout: home

hero:
  text: "EvoLinear.jl"
  tagline: Regularized linear models for tabular data
  image:
    src: /logo.png
    alt: Evovest
  
  actions:
    - theme: brand
      text: Quick start
      link: /quick-start
    - theme: alt
      text: Design
      link: /design
    - theme: alt
      text: Models
      link: /models
    - theme: alt
      text: Tutorials
      link: /tutorials-logistic-titanic.md
    - theme: alt
      text: Source code
      link: https://github.com/jeremiedb/EvoLinear.jl
---
```
<<<<<<< HEAD
=======
pkg> add https://github.com/jeremiedb/EvoLinear.jl
```

## Getting started

Build a configuration struct with `EvoLinearRegressor`. Then `EvoLinear.fit` takes `x::Matrix` and `y::Vector` as inputs, plus optionally `w::Vector` as weights and fits a linear boosted model.

```julia
using EvoLinear
config = EvoLinearRegressor(loss=:mse, L1=1e-1, L2=1e-2, nrounds=10)
m = EvoLinear.fit(config; x, y, metric=:mse)
p = m(x)
```
>>>>>>> main
