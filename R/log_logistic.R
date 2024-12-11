
#' Fit log-logistic model
#'
#' @param data_vr input data counts
#' @param data_direct input data in pre-processed form
#' @param time_model pspline or rw2
#' @param start_year first year to estimate
#' @param end_year last year to estimate
#' @param include_iid_in_pred include IID errors in prediction
#'
#' @return
#' @export
#' @importFrom stats nlminb
#'
#' @examples
log_logistic <- function(data_vr, data_direct, time_model, start_year = 1950, end_year = 2030, include_iid_in_pred = F) {
  # add: data_fbh, data_other
  
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
                      pop = data_vr$pop)
    param_list <- list(intercept_log_shape = 0,
                       intercept_log_scale = 0,
                       log_tau_delta_log_shape = 0,
                       log_tau_delta_log_scale = 0,
                       #log_phi = 0,
                       log_tau_epsilon = 0,
                       delta_log_shape = rep(0, n_betas),
                       delta_log_scale = rep(0, n_betas),
                       epsilon = rep(0, data_vr$n_obs_vr))
    random <- c("delta_log_shape", "delta_log_scale",
                "epsilon")
    
    init_lower <- c(-1, 2, -10, -15, -10, -10, -5)
    init_upper <- c(2, 250, 10, 10, 20, 20, 5)

  } else if (time_model == "rw2") {
    
    inla.rw = utils::getFromNamespace("inla.rw", "INLA")
    R <- inla.rw(n_years, order = 2, sparse = T, scale.model = T)
    R <- R + diag(n_years) * 1e-6
    
    data_list <- list(model = "ll_vr_rw",
                      n_years = n_years,
                      R = R,
                      n_obs_vr = data_vr$n_obs_vr,
                      time_id_vr = data_vr$time_id_vr,
                      months_vr = data_vr$months_vr,
                      obs_vr = data_vr$obs_vr,
                      n_vr = data_vr$n_vr,
                      births = data_vr$births,
                      pop = data_vr$pop) #,
                      # n_obs_direct = data_direct$n_obs_direct,
                      # time_id_direct = data_direct$time_id_direct,
                      # months_direct = data_direct$months_direct,
                      # obs_direct = data_direct$obs_direct,
                      # se_direct = data_direct$se_direct)
    param_list <- list(intercept_log_shape = 0.7, #-1.5,
                       intercept_log_scale = 39,
                       log_tau_delta_log_shape = 0,
                       log_tau_delta_log_scale = 0,
                       log_tau_epsilon_log_shape = 0,
                       log_tau_epsilon_log_scale = 0,
                       # epsilon = rep(0, data_vr$n_obs_vr),
                       # log_prec_epsilon = 9,
                       # log_phi = 0,
                       delta_log_shape = rep(0, n_years),
                       delta_log_scale = rep(0, n_years),
                       epsilon_log_shape = rep(0, n_years),
                       epsilon_log_scale = rep(0, n_years))
    random <- c("delta_log_shape", "delta_log_scale",
               "epsilon_log_shape", "epsilon_log_scale")
    
    init_lower <- c(-1, 2, -10, -15, -10, -10)
    init_upper <- c(2, 250, 10, 10, 20, 20)
      
  } else {
     stop("Invalid time model.")
  }
  
  obj <- TMB::MakeADFun(data = data_list,
                        parameters = param_list,
                        random = random,
                        map = list(),
                        hessian = TRUE,
                        DLL = "childSurvLL_TMBExports")
  
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
  intercept.shape.idx <- which(names(mu) == "intercept_log_shape")
  intercept.scale.idx <- which(names(mu) == "intercept_log_scale")
  time.struct.shape.idx <- which(names(mu) == "delta_log_shape")
  time.struct.scale.idx <- which(names(mu) == "delta_log_scale")
  time.unstruct.shape.idx <- which(names(mu) == "epsilon_log_shape")
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
    
    # combine draws for linear predictor (log shape)
    fitted_beta_shape <-
      matrix(rep(t.draws[intercept.shape.idx,], n_betas), nrow = n_betas, byrow = T) +
      t.draws[time.struct.shape.idx,]
    fitted_shape <- B %*% fitted_beta_shape
    
    # combine draws for linear predictor (log scale)
    fitted_beta_scale <-
      matrix(rep(t.draws[intercept.scale.idx,], n_betas), nrow = n_betas, byrow = T) +
      t.draws[time.struct.scale.idx,]
    fitted_scale <- B %*% fitted_beta_scale
    
    # log_phi draws
    #fitted_log_phi <- t.draws[log.phi.idx,]
    # overdispersion draws
    fitted_overdispersion <- t.draws[which(names(mu) == "log_tau_epsilon"),]
    
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
    
    # combine draws for linear predictor (log shape)
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
    nmr_mat[t,] <- flexsurv::pllogis(1, exp(-1*exp(fitted_shape))[t,], exp(fitted_scale)[t,])
    imr_mat[t,] <- flexsurv::pllogis(12, exp(-1*exp(fitted_shape))[t,], exp(fitted_scale)[t,])
    u5mr_mat[t,] <- flexsurv::pllogis(60, exp(-1*exp(fitted_shape))[t,], exp(fitted_scale)[t,])
  }
  
  #  create prediction data.frame
  df_pred <- data.frame(year = start_year:end_year)
  
  # get summaries
  df_pred$log_shape_smoothed_med <- apply(fitted_shape, 1, quantile, 0.5)
  df_pred$log_shape_smoothed_lower <- apply(fitted_shape, 1, quantile, 0.05)
  df_pred$log_shape_smoothed_upper <- apply(fitted_shape, 1, quantile, 0.95)
  df_pred$log_scale_smoothed_med <- apply(fitted_scale, 1, quantile, 0.5)
  df_pred$log_scale_smoothed_lower <- apply(fitted_scale, 1, quantile, 0.05)
  df_pred$log_scale_smoothed_upper <- apply(fitted_scale, 1, quantile, 0.95)
  df_pred$nmr_smoothed_med <- apply(nmr_mat, 1, quantile, 0.5)
  df_pred$nmr_smoothed_lower <- apply(nmr_mat, 1, quantile, 0.05)
  df_pred$nmr_smoothed_upper <- apply(nmr_mat, 1, quantile, 0.95)
  df_pred$imr_smoothed_med <- apply(imr_mat, 1, quantile, 0.5)
  df_pred$imr_smoothed_lower <- apply(imr_mat, 1, quantile, 0.05)
  df_pred$imr_smoothed_upper <- apply(imr_mat, 1, quantile, 0.95)
  df_pred$u5mr_smoothed_med <- apply(u5mr_mat, 1, quantile, 0.5)
  df_pred$u5mr_smoothed_lower <- apply(u5mr_mat, 1, quantile, 0.05)
  df_pred$u5mr_smoothed_upper <- apply(u5mr_mat, 1, quantile, 0.95)
  
  #return(list(df_pred, fitted_log_phi))
  #return(df_pred)
  return(list(df_pred, fitted_overdispersion))
}