# childSurv

Functions to fit survival model for child mortality over time, using full birth history, vital registration, and/or pre-processed mortality rate estimates. This package uses Template Model Builder for computation.

This package produces estimates as described by: Paulson, K. R., Okonek, T., \& Wakefield, J. (2026). A Survival Framework for Estimating Child Mortality Rates using Multiple Data Types. arXiv preprint arXiv:2601.20821.

## Installation

This package can be installed using `devtools::install_github()`:

```R
devtools::install_github(repo="krpaulson/childSurv")
```

## Steps

1. If using full birth history microdata, first use [pssst](https://github.com/taylorokonek/pssst) package to process into estimates of survival parameters. Follow instructions provided by that package.
2. Use `mm_adjustment` function to apply missing mothers adjustment for countries with large HIV epidemics.
3. If using vital registration data or pre-processed estimates, prep data according to help documentation for `log_logistic` and `piecewise_exponential` functions.
4. Run `log_logistic` and/or `piecewise_exponential` function, depending on which parametric survival function you choose.
