
#' Fit log-logistic model
#' 
#' @description Fit log-logistic survival model to input data which may
#'  include any combination of pseudo-likelihood estimates from
#'  full birth histories, vital registration counts (deaths, births,
#'  mid-year population), and pre-processed estimates of mortality rates. The
#'  temporal model can be either a penalized spline or a second order random
#'  walk. When VR is included, the count model can be either Poisson or
#'  Poisson-lognormal to capture overdispersion.
#'
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
#'      belongs to (1=shape; 2=scale).}
#'  }
#' @param data_vr Input data with VR counts. If not NULL, is a list with items:
#'  \describe{
#'    \item{n_obs_vr}{Number of observations from vital registration}
#'    \item{time_id_vr}{Vector containing numeric indicator for which time point
#'      each observation belongs to within the vector start_year:end_year}
#'    \item{months_vr}{Data are for nDx -- this vector input tells us what x is
#'      in months}
#'    \item{n_vr}{Data are for nDx -- this vector input tells us what n is in
#'      months}
#'    \item{obs_vr}{Vector containing observations of nDx (death counts)}
#'    \item{births}{Vector containing births in year of observation. Can be
#'      NA for entries corresponding to age groups that aren't neonatal.}
#'    \item{pop}{Vector containing mid-year population in the age group and
#'      year corresponding to each observation. Can be NA for entries
#'      corresponding to neonatal death counts.}
#'  }
#' @param data_pp Input data in pre-processed mortality rate form. If not NULL,
#'  is a list with the items:
#'  \describe{
#'    \item{n_obs_pp}{Number of pre-processed estimates}
#'    \item{time_id_pp}{Vector containing numeric indicator for which time point
#'      each observation belongs to within the vector start_year:end_year}
#'    \item{months_pp}{Pre-processed estimates are for logit(nq0) -- this 
#'      vector input tells us what n is in months.}
#'    \item{obs_pp}{Pre-processed estimate of logit(nq0)}
#'    \item{se_pp}{Standard error of pre-processed estimate of logit(nq0)}
#'  }
#' @param time_model One of either 'pspline' or 'rw2'
#' @param count_model One of either 'poisson' or 'poisson_lognormal'. If no
#'    VR data this argument does not impact the results.
#' @param start_year First year to estimate
#' @param end_year Last year to estimate
#' @param include_iid_in_pred Whether to include IID errors in prediction
#'
#' @return For poisson count model, returns data.frame with predictions only.
#'  For poisson lognormal, returns prediction data frame in addition to fitted
#'  overdispersion and fitted log precision for shape and scale parameters.
#'  
#' @export
#' @importFrom stats nlminb
log_logistic <- function(data_fbh = NULL, data_vr = NULL, data_pp = NULL, 
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

    data_list <- list(model = "ll_vr_pspline",
                      B=B, A=A, R=R,
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
      param_list <- list(intercept_logit_shape = 0.7,
                         intercept_log_scale = 39,
                         log_tau_delta_logit_shape = -1,
                         log_tau_delta_log_scale = -7,
                         delta_logit_shape = rep(0, n_betas),
                         delta_log_scale = rep(0, n_betas))
      random <- c("delta_logit_shape", "delta_log_scale")
      init_lower <- c(-1, 2, -2, -10)
      init_upper <- c(2, 250, 16, 16)
      
    } else if (count_model == "poisson_lognormal") {
      data_list$model <- "ll_vr_pspline_poisson_lognormal"
      param_list <- list(intercept_logit_shape = 0.7,
                         intercept_log_scale = 39,
                         log_tau_delta_logit_shape = -1,
                         log_tau_delta_log_scale = -7,
                         log_tau_epsilon = 0,
                         delta_logit_shape = rep(0, n_betas),
                         delta_log_scale = rep(0, n_betas),
                         epsilon = rep(0, data_vr$n_obs_vr))
      random <- c("delta_logit_shape", "delta_log_scale", "epsilon")
      init_lower <- c(-1, 2, -2, -10, -5)
      init_upper <- c(2, 250, 16, 16, 5)
    }

  } else if (time_model == "rw2") {
    
    inla.rw = utils::getFromNamespace("inla.rw", "INLA")
    R <- inla.rw(n_years, order = 2, sparse = T, scale.model = T)
    R <- R + diag(n_years) * 1e-6
    
    data_list <- list(
      model = "ll_rw",
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
    param_list <- list(
      intercept_logit_shape = -1.2,
      intercept_log_scale = 23,
      log_tau_delta_logit_shape = 0,
      log_tau_delta_log_scale = 0,
      log_tau_epsilon_logit_shape = 0,
      log_tau_epsilon_log_scale = 0,
      delta_logit_shape = rep(0, n_years),
      delta_log_scale = rep(0, n_years),
      epsilon_logit_shape = rep(0, n_years),
      epsilon_log_scale = rep(0, n_years)
    )
    random <- c("delta_logit_shape", "delta_log_scale",
               "epsilon_logit_shape", "epsilon_log_scale")
    init_lower <- c(-5, 0.1, -10, -15, -10, -10)
    init_upper <- c(2, 250, 10, 10, 20, 20)
      
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
  intercept.shape.idx <- which(names(mu) == "intercept_logit_shape")
  intercept.scale.idx <- which(names(mu) == "intercept_log_scale")
  time.struct.shape.idx <- which(names(mu) == "delta_logit_shape")
  time.struct.scale.idx <- which(names(mu) == "delta_log_scale")
  time.unstruct.shape.idx <- which(names(mu) == "epsilon_logit_shape")
  time.unstruct.scale.idx <- which(names(mu) == "epsilon_log_scale")
  log.phi.idx <- which(names(mu) == "log_phi")
  
  if (time_model == "pspline") {
    
    # create list of constraint matrices for these terms
    A.mat.list <- list()
    A.mat.list[[1]] <- matrix(1, nrow = 1, ncol = n_betas)
    A.mat.list[[2]] <- matrix(1, nrow = 1, ncol = n_betas)
    
    # sample
    # Reference: https://github.com/taylorokonek/stbench/blob/main/R/multiconstr_prec.R
    multiconstr_prec = utils::getFromNamespace("multiconstr_prec", "stbench")
    t.draws <- multiconstr_prec(mu = mu,
                                prec = SD0$jointPrecision,
                                n.sims = 1000,
                                constrain.idx.list = list(time.struct.shape.idx,
                                                          time.struct.scale.idx),
                                A.mat.list = A.mat.list)
    
    # take the constrained draws
    t.draws <- t.draws$x.c
    
    # combine draws for linear predictor (logit shape)
    fitted_beta_shape <-
      matrix(rep(t.draws[intercept.shape.idx,], n_betas), nrow = n_betas, byrow = T) +
      t.draws[time.struct.shape.idx,]
    fitted_shape <- B %*% fitted_beta_shape
    
    # combine draws for linear predictor (log scale)
    fitted_beta_scale <-
      matrix(rep(t.draws[intercept.scale.idx,], n_betas), nrow = n_betas, byrow = T) +
      t.draws[time.struct.scale.idx,]
    fitted_scale <- B %*% fitted_beta_scale
    
    # overdispersion draws
    if (count_model == "poisson_lognormal") {
      fitted_overdispersion <- t.draws[which(names(mu) == "log_tau_epsilon"),]
      fitted_log_tau_shape <- t.draws[which(names(mu) == "log_tau_delta_logit_shape"),]
      fitted_log_tau_scale <- t.draws[which(names(mu) == "log_tau_delta_log_scale"),]
    }
    
  } else if(time_model == "rw2") {
    
    # create list of constraint matrices for these terms
    A.mat.list <- list()
    A.mat.list[[1]] <- matrix(1, nrow = 1, ncol = n_years)
    A.mat.list[[2]] <- matrix(1, nrow = 1, ncol = n_years)
    
    # sample
    # Reference: https://github.com/taylorokonek/stbench/blob/main/R/multiconstr_prec.R
    multiconstr_prec = utils::getFromNamespace("multiconstr_prec", "stbench")
    t.draws <- multiconstr_prec(mu = mu,
                                prec = SD0$jointPrecision,
                                n.sims = 1000,
                                constrain.idx.list = list(time.struct.shape.idx,
                                                          time.struct.scale.idx),
                                A.mat.list = A.mat.list)
    
    # take the constrained draws
    t.draws <- t.draws$x.c
    
    # combine draws for linear predictor (logit shape)
    fitted_shape <-
      matrix(rep(t.draws[intercept.shape.idx,], n_years), nrow = n_years, byrow = T) +
      t.draws[time.struct.shape.idx,]
    
    # combine draws for linear predictor (log scale)
    fitted_scale <-
      matrix(rep(t.draws[intercept.scale.idx,], n_years), nrow = n_years, byrow = T) +
      t.draws[time.struct.scale.idx,]
    
    # add IID
    if (include_iid_in_pred) {
      fitted_shape <- fitted_shape + t.draws[time.unstruct.shape.idx,]
      fitted_scale <- fitted_scale + t.draws[time.unstruct.scale.idx,]
    }
    
  } else {
    stop("Invalid temporal model.")
  }
  
  # combine to get nmr, imr, u5mr
  nmr_mat <- matrix(nrow = n_years, ncol = 1000)
  imr_mat <- matrix(nrow = n_years, ncol = 1000)
  u5mr_mat <- matrix(nrow = n_years, ncol = 1000)
  for (t in 1:n_years) {
    nmr_mat[t,] <- flexsurv::pllogis(1, plogis(fitted_shape[t,]), exp(fitted_scale)[t,])
    imr_mat[t,] <- flexsurv::pllogis(12, plogis(fitted_shape[t,]), exp(fitted_scale)[t,])
    u5mr_mat[t,] <- flexsurv::pllogis(60, plogis(fitted_shape[t,]), exp(fitted_scale)[t,])
  }
  
  # helper function to compute mean/lower/upper and return a data.frame
  # this does operations like apply(fitted_shape, 1, quantile, 0.5) for us
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
    quantile_summary(fitted_shape, "logit_shape"),
    quantile_summary(fitted_scale, "log_scale"),
    quantile_summary(nmr_mat, "nmr"),
    quantile_summary(imr_mat, "imr"),
    quantile_summary(u5mr_mat, "u5mr")
  )
  
  if (count_model == "poisson") {
    return(df_pred)
  } else if (count_model == "poisson_lognormal") {
    return(list(df_pred, fitted_overdispersion, fitted_log_tau_shape, fitted_log_tau_scale))
  }
}