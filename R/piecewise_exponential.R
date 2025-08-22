
#' Fit piecewise-exponential model
#' 
#' @description Fit piecewise-exponential survival model with breakpoints at
#'  1 and 12 months to input data which may include any combination of
#'  pseudo-likelihood estimates from full birth histories, vital registration
#'  counts (deaths, births, mid-year population), and pre-processed estimates
#'  of mortality rates. The temporal model can be either a penalized spline or
#'  a second order random walk. When VR is included, the count model can be
#'  either Poisson or Poisson-lognormal to capture overdispersion.
#'
#' @inheritParams log_logistic
#' @param data_fbh Input data pseudo-likelihood estimates from full birth
#'  histories. If not NULL, is a list with items:
#'  \describe{
#'    \item{n_obs_fbh}{Number of observations (number of surveys times number
#'      of years per survey). Also the length of thetahat.}
#'    \item{thetahat}{Vector of pseudo-likelihood observations.}
#'    \item{V}{Variance-covariance matrix for all pseudo-likelihood estimates.}
#'    \item{time_id_fbh}{Vector containing numeric indicator for which time point
#'      each observation belongs to within the vector start_year:end_year}
#'    \item{par_id_fbh}{Numeric indicator for which parameter an observation
#'      belongs to (1=alpha1, 2=alpha2, 3=alpha3).}
#'  }
#'
#' @return For poisson count model, returns data.frame with predictions only.
#'  For poisson lognormal, returns prediction data frame in addition to fitted
#'  overdispersion and fitted log precision for alpha1, alpha2, and alpha3
#'  parameters.
#'  
#' @export
#' @importFrom stats nlminb
piecewise_exponential <- function(data_fbh = NULL, data_vr = NULL, data_pp = NULL, 
                                  time_model = "rw2", count_model = "poisson",
                                  start_year = 1950, end_year = 2030,
                                  include_iid_in_pred = FALSE) {
  
  # TODO: revisit p-spline
  if (time_model != "rw2") stop("Only rw2 supported right now.")
  
  # TODO: revisit poisson-lognormal
  if (count_model != "poisson") stop("Only poisson supported right now.")
  
  # if any data types are omitted, set to acceptable values for TMB
  if (is.null(data_fbh)) {
    data_fbh <- list(
      n_obs_fbh = 0,
      thetahat = NA_real_,
      V = matrix(NA_real_),
      time_id_fbh = NA_integer_,
      par_id_fbh = NA_integer_
    )
  }
  if (is.null(data_vr)) {
    data_vr <- list(
      n_obs_vr = 0,
      time_id_vr = NA_integer_,
      months_vr = NA_real_,
      n_vr = NA_real_,
      obs_vr = NA_real_,
      births = NA_real_,
      pop = NA_real_
    )
  }
  if (is.null(data_pp)) {
    data_pp <- list(
      n_obs_pp = 0,
      time_id_pp = NA_integer_,
      months_pp = NA_real_,
      obs_pp = NA_real_,
      se_pp = NA_real_
    )
  }
  
  # check that no time IDs are NA
  if (anyNA(data_fbh$time_id_fbh) & !all(is.na(data_fbh$time_id_fbh))) {
    stop("Some FBH time id values are NA")
  }
  if (anyNA(data_vr$time_id_vr) & !all(is.na(data_vr$time_id_vr))) {
    stop("Some VR time id values are NA")
  }
  if (anyNA(data_pp$time_id_pp) & !all(is.na(data_pp$time_id_pp))) {
    stop("Some pre-processed time id values are NA")
  }
  
  # check that time_model and count_model are valid.
  if (!time_model %in% c("pspline", "rw2")) {
    stop("`time_model` must be one of 'pspline' or 'rw2'.")
  }
  if (!count_model %in% c("poisson", "poisson_lognormal") &
      data_vr$n_obs_vr > 0) {
    stop("`count_model` must be one of 'poisson' or 'poisson_lognormal'.")
  }
  
  # check that pre-processed estimates have non-NA standard error
  if (data_pp$n_obs_pp > 0 & anyNA(data_pp$se_pp)) {
    stop("data_pp$se_pp contains NA values.")
  }
  
  n_years <- length(start_year:end_year)
  
  if (time_model == "pspline") {

    B <- bspline(1:n_years)
    n_betas <- ncol(B)
    A <- matrix(1, nrow=1, ncol=n_betas)
    inla.rw <- utils::getFromNamespace("inla.rw", "INLA")
    R <- inla.rw(n_betas, order = 2, sparse = T, scale.model = T)
    R <- R + diag(n_betas) * 1e-6
    
    data_list <- list(model = "pe_vr_pspline",
                      B = B, A = A, R = R,
                      n_obs_vr = data_vr$n_obs_vr,
                      time_id_vr = data_vr$time_id_vr,
                      months_vr = data_vr$months_vr,
                      obs_vr = data_vr$obs_vr,
                      n_vr = data_vr$n_vr,
                      births = data_vr$births,
                      pop = data_vr$pop,
                      n_obs_direct = data_direct$n_obs_direct,
                      time_id_direct = data_direct$time_id_direct,
                      months_direct = data_direct$months_direct,
                      obs_direct = data_direct$obs_direct,
                      se_direct = data_direct$se_direct)
    
    if (count_model == "poisson") {
      param_list <- list(intercept_log_alpha0 = -10.0,
                         intercept_log_alpha1 = -8.4,
                         intercept_log_alpha2 = -5.1,
                         log_tau_delta_log_alpha0 = 1.4,
                         log_tau_delta_log_alpha1 = -2.0,
                         log_tau_delta_log_alpha2 = -0.17,
                         delta_log_alpha0 = rep(0, n_betas),
                         delta_log_alpha1 = rep(0, n_betas),
                         delta_log_alpha2 = rep(0, n_betas))
      random <- c("delta_log_alpha0", "delta_log_alpha1", "delta_log_alpha2")
      init_lower <- c(-20, -20, -20, -20, -20, -20)
      init_upper <- c(10, 10, 10, 10, 10, 10)
      
    } else if (count_model == "poisson_lognormal") {
      data_list$model <- "pe_vr_pspline_pois_lognormal"
      param_list <- list(intercept_log_alpha0 = -10.0,
                         intercept_log_alpha1 = -8.4,
                         intercept_log_alpha2 = -5.1,
                         log_tau_delta_log_alpha0 = 1.4,
                         log_tau_delta_log_alpha1 = -2.0,
                         log_tau_delta_log_alpha2 = -0.17,
                         log_tau_epsilon = 0,
                         delta_log_alpha0 = rep(0, n_betas),
                         delta_log_alpha1 = rep(0, n_betas),
                         delta_log_alpha2 = rep(0, n_betas),
                         epsilon = rep(0, data_vr$n_obs_vr))
      random <- c("delta_log_alpha0", "delta_log_alpha1", "delta_log_alpha2", "epsilon")
      init_lower <- c(-20, -20, -20, -20, -20, -20, -20)
      init_upper <- c(10, 10, 10, 10, 10, 10, 10)
    }
    
  } else if (time_model == "rw2") {
    
    inla.rw = utils::getFromNamespace("inla.rw", "INLA")
    R <- inla.rw(n_years, order = 2, sparse = T, scale.model = T)
    R <- R + diag(n_years) * 1e-6
    
    data_list <- list(
      model = "pe_rw",
      n_years = n_years,
      R = R,
      # ---- FBH
      n_obs_fbh = data_fbh$n_obs_fbh,
      thetahat = data_fbh$thetahat,
      V = data_fbh$V,
      time_id_fbh = data_fbh$time_id_fbh,
      par_id_fbh = data_fbh$par_id_fbh,
      # ---- VR
      n_obs_vr = data_vr$n_obs_vr,
      time_id_vr = data_vr$time_id_vr,
      months_vr = data_vr$months_vr,
      n_vr = data_vr$n_vr,
      obs_vr = data_vr$obs_vr,
      births = data_vr$births,
      pop = data_vr$pop,
      # ---- Pre-processed
      n_obs_pp = data_pp$n_obs_pp,
      time_id_pp = data_pp$time_id_pp,
      months_pp = data_pp$months_pp,
      obs_pp = data_pp$obs_pp,
      se_pp = data_pp$se_pp
    )
    param_list <- list(intercept_log_alpha1 = -4,
                       intercept_log_alpha2 = -4,
                       intercept_log_alpha3 = -4,
                       log_tau_delta_log_alpha1 = 0,
                       log_tau_delta_log_alpha2 = 0,
                       log_tau_delta_log_alpha3 = 0,
                       log_tau_epsilon_log_alpha1 = 0,
                       log_tau_epsilon_log_alpha2 = 0,
                       log_tau_epsilon_log_alpha3 = 0,
                       delta_log_alpha1 = rep(0, n_years),
                       delta_log_alpha2 = rep(0, n_years),
                       delta_log_alpha3 = rep(0, n_years),
                       epsilon_log_alpha1 = rep(0, n_years),
                       epsilon_log_alpha2 = rep(0, n_years),
                       epsilon_log_alpha3 = rep(0, n_years))
    random = c("delta_log_alpha1", "delta_log_alpha2", "delta_log_alpha3",
               "epsilon_log_alpha1", "epsilon_log_alpha2", "epsilon_log_alpha3")
    
    init_lower <- c(-15, -15, -15, -20, -20, -20, -20, -20, -20)
    init_upper <- c(5, 5, 5, 20, 20, 20, 20, 20, 20)

  } else {
    stop("Invalid time model.")
  }
  
  obj <- TMB::MakeADFun(data = data_list,
                        parameters = param_list,
                        random = random,
                        map = list(),
                        hessian = TRUE,
                        DLL = "childSurv_TMBExports")
  opt <- nlminb(obj$par, obj$fn, obj$gr,
                lower = init_lower, upper = init_upper)
  SD0 <- TMB::sdreport(obj,
                       getJointPrecision = TRUE,
                       getReportCovariance = TRUE,
                       bias.correct = TRUE,
                       bias.correct.control = list(sd = TRUE))
  
  
  # TMB post-processing ------------------------------------------------------
  
  # obtain point estimates (means) for fixed and random effects
  mu <- c(SD0$par.fixed, SD0$par.random)
  
  # get ids for different parameters
  intercept.alpha1.idx <- which(names(mu) == "intercept_log_alpha1")
  intercept.alpha2.idx <- which(names(mu) == "intercept_log_alpha2")
  intercept.alpha3.idx <- which(names(mu) == "intercept_log_alpha3")
  time.struct.alpha1.idx <- which(names(mu) == "delta_log_alpha1")
  time.struct.alpha2.idx <- which(names(mu) == "delta_log_alpha2")
  time.struct.alpha3.idx <- which(names(mu) == "delta_log_alpha3")
  time.unstruct.alpha1.idx <- which(names(mu) == "epsilon_log_alpha1")
  time.unstruct.alpha2.idx <- which(names(mu) == "epsilon_log_alpha2")
  time.unstruct.alpha3.idx <- which(names(mu) == "epsilon_log_alpha3")
  
  if (time_model == "pspline") {
    
    # create list of constraint matrices for these terms
    A.mat.list <- list()
    A.mat.list[[1]] <- matrix(1, nrow = 1, ncol = n_betas)
    A.mat.list[[2]] <- matrix(1, nrow = 1, ncol = n_betas)
    A.mat.list[[3]] <- matrix(1, nrow = 1, ncol = n_betas)
    
    # sample
    # Reference: https://github.com/taylorokonek/stbench/blob/main/R/multiconstr_prec.R
    multiconstr_prec = utils::getFromNamespace("multiconstr_prec", "stbench")
    t.draws <- multiconstr_prec(mu = mu,
                                prec = SD0$jointPrecision,
                                n.sims = 1000,
                                constrain.idx.list = list(time.struct.alpha0.idx,
                                                          time.struct.alpha1.idx,
                                                          time.struct.alpha2.idx),
                                A.mat.list = A.mat.list)
    
    # take the constrained draws
    t.draws <- t.draws$x.c
    
    # combine draws for linear predictor (alpha0)
    fitted_alpha0 <-
      matrix(rep(t.draws[intercept.alpha0.idx,], n_betas), nrow = n_betas, byrow = T) +
      t.draws[time.struct.alpha0.idx,]
    fitted_alpha0 <- B %*% fitted_alpha0
    
    # combine draws for linear predictor (alpha1)
    fitted_alpha1 <-
      matrix(rep(t.draws[intercept.alpha1.idx,], n_betas), nrow = n_betas, byrow = T) +
      t.draws[time.struct.alpha1.idx,]
    fitted_alpha1 <- B %*% fitted_alpha1
    
    # combine draws for linear predictor (alpha2)
    fitted_alpha2 <-
      matrix(rep(t.draws[intercept.alpha2.idx,], n_betas), nrow = n_betas, byrow = T) +
      t.draws[time.struct.alpha2.idx,]
    fitted_alpha2 <- B %*% fitted_alpha2
    
  } else if(time_model == "rw2") {
    
    # create list of constraint matrices for these terms
    A.mat.list <- list()
    A.mat.list[[1]] <- matrix(1, nrow = 1, ncol = n_years)
    A.mat.list[[2]] <- matrix(1, nrow = 1, ncol = n_years)
    A.mat.list[[3]] <- matrix(1, nrow = 1, ncol = n_years)
    
    # sample
    # Reference: https://github.com/taylorokonek/stbench/blob/main/R/multiconstr_prec.R
    multiconstr_prec = utils::getFromNamespace("multiconstr_prec", "stbench")
    t.draws <- multiconstr_prec(mu = mu,
                                prec = SD0$jointPrecision,
                                n.sims = 1000,
                                constrain.idx.list = list(time.struct.alpha1.idx,
                                                          time.struct.alpha2.idx,
                                                          time.struct.alpha3.idx),
                                A.mat.list = A.mat.list)
    
    # take the constrained draws
    t.draws <- t.draws$x.c
    
    # combine draws for linear predictor (logit alpha1)
    fitted_alpha1 <-
      matrix(rep(t.draws[intercept.alpha1.idx,], n_years), nrow = n_years, byrow = T) +
      t.draws[time.struct.alpha1.idx,]
    
    # combine draws for linear predictor (logit alpha2)
    fitted_alpha2 <-
      matrix(rep(t.draws[intercept.alpha2.idx,], n_years), nrow = n_years, byrow = T) +
      t.draws[time.struct.alpha2.idx,]
    
    # combine draws for linear predictor (logit alpha3)
    fitted_alpha3 <-
      matrix(rep(t.draws[intercept.alpha3.idx,], n_years), nrow = n_years, byrow = T) +
      t.draws[time.struct.alpha3.idx,]
    
    # add IID
    if (include_iid_in_pred) {
      fitted_alpha1 <- fitted_alpha1 + t.draws[time.unstruct.alpha1.idx,]
      fitted_alpha2 <- fitted_alpha2 + t.draws[time.unstruct.alpha2.idx,]
      fitted_alpha3 <- fitted_alpha3 + t.draws[time.unstruct.alpha3.idx,]
    }
    
  } else {
    stop("Invalid temporal model.")
  }
  
  # combine to get nmr, imr, u5mr
  nmr_mat <- matrix(nrow = n_years, ncol = 1000)
  imr_mat <- matrix(nrow = n_years, ncol = 1000)
  u5mr_mat <- matrix(nrow = n_years, ncol = 1000)
  
  for (t in 1:n_years) {
    nmr_mat[t,] <- 1-exp(-1 * (exp(fitted_alpha1)[t,] + exp(fitted_alpha2)[t,] + exp(fitted_alpha3)[t,]))
    imr_mat[t,] <- 1-exp(-1 * (12*exp(fitted_alpha1)[t,] + 12*exp(fitted_alpha2)[t,] + exp(fitted_alpha3)[t,]))
    u5mr_mat[t,] <- 1-exp(-1 * (60*exp(fitted_alpha1)[t,] + 12*exp(fitted_alpha2)[t,] + exp(fitted_alpha3)[t,]))
  }
  
  # helper function to compute mean/lower/upper and return a data.frame
  # this does operations like apply(fitted_alpha1, 1, quantile, 0.5) for us
  quantile_summary <- function(mat, name) {
    probs <- c(0.5, 0.05, 0.95)
    labels <- c("med", "lower", "upper")
    stats <- lapply(probs, function(p) apply(mat, 1, quantile, probs = p))
    names(stats) <- paste0(name, "_smoothed_", labels)
    as.data.frame(stats)
  }
  
  # create prediction data.frame with summaries
  df_pred <- cbind(
    data.frame(year = start_year:end_year),
    quantile_summary(fitted_alpha1, "log_alpha1"),
    quantile_summary(fitted_alpha2, "log_alpha2"),
    quantile_summary(fitted_alpha3, "log_alpha3"),
    quantile_summary(nmr_mat, "nmr"),
    quantile_summary(imr_mat, "imr"),
    quantile_summary(u5mr_mat, "u5mr")
  )
  
  if (count_model == "poisson") {
    return(df_pred)
  } else if (count_model == "poisson_lognormal") {
    return(list(df_pred, fitted_overdispersion, fitted_log_tau_alpha1, fitted_log_tau_alpha2,
                fitted_log_tau_alpha3))
  }
}