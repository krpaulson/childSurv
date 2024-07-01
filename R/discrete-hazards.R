
#' Fit log-logistic model
#'
#' @param data_vr input data counts
#' @param data_direct input data in pre-processed form
#' @param time_model spline or rw2
#' @param start_year first year to estimate
#' @param end_year last year to estimate
#' @param include_iid_in_pred include IID errors in prediction
#'
#' @return
#' @export
#' @importFrom stats nlminb
#'
#' @examples
discrete_hazards <- function(data_vr, data_direct, time_model, start_year = 1950, end_year = 2030, include_iid_in_pred = F) {
  # add: data_fbh, data_other
  
  n_years <- length(start_year:end_year)
  
  # if (time_model == "spline") {
  #   
  #   B <- bspline(1:n_years)
  #   n_betas <- ncol(B)
  #   A <- matrix(1, nrow=1, ncol=n_betas)
  #   inla.rw <- utils::getFromNamespace("inla.rw", "INLA")
  #   R <- inla.rw(n_betas, order = 2, sparse = T, scale.model = T)
  #   R <- R + diag(n_betas) * 1e-6
  #   
  #   data_list <- list(model = "ll_vr_rw",
  #                     B=B, A=A, R=R,
  #                     n_obs_vr = data_vr$n_obs_vr,
  #                     time_id_vr = data_vr$time_id_vr,
  #                     months_vr = data_vr$months_vr,
  #                     obs_vr = data_vr$obs_vr,
  #                     se_vr = data_vr$se_vr)
  #   param_list <- list(intercept_log_shape = 0,
  #                      intercept_log_scale = 0,
  #                      log_tau_delta_log_shape = 0,
  #                      log_tau_delta_log_scale = 0,
  #                      delta_log_shape = rep(0, n_betas),
  #                      delta_log_scale = rep(0, n_betas))
  #   random <- c("delta_log_shape", "delta_log_scale")
  #   
  # } else if (time_model == "rw2") {
    
    inla.rw = utils::getFromNamespace("inla.rw", "INLA")
    R <- inla.rw(n_years, order = 2, sparse = T, scale.model = T)
    R <- R + diag(n_years) * 1e-6
    
    data_list <- list(model = "dh_vr_rw",
                      n_years = n_years,
                      R = R,
                      n_obs_vr = data_vr$n_obs_vr,
                      time_id_vr = data_vr$time_id_vr,
                      months_vr = data_vr$months_vr,
                      obs_vr = data_vr$obs_vr,
                      n_vr = data_vr$n_vr,
                      births = data_vr$births,
                      pop = data_vr$pop)
    param_list <- list(intercept_logit_B0 = -4,
                       intercept_logit_B1 = -4,
                       intercept_logit_B2 = -4,
                       log_tau_delta_logit_B0 = 0,
                       log_tau_delta_logit_B1 = 0,
                       log_tau_delta_logit_B2 = 0,
                       log_tau_epsilon_logit_B0 = 0,
                       log_tau_epsilon_logit_B1 = 0,
                       log_tau_epsilon_logit_B2 = 0,
                       #log_phi = 0,
                       delta_logit_B0 = rep(0, n_years),
                       delta_logit_B1 = rep(0, n_years),
                       delta_logit_B2 = rep(0, n_years),
                       epsilon_logit_B0 = rep(0, n_years),
                       epsilon_logit_B1 = rep(0, n_years),
                       epsilon_logit_B2 = rep(0, n_years))
    random = c("delta_logit_B0", "delta_logit_B1", "delta_logit_B2",
               "epsilon_logit_B0", "epsilon_logit_B1", "epsilon_logit_B2")
      
  # } else {
  #   stop("Invalid time model.")
  # }
  
  obj <- TMB::MakeADFun(data = data_list,
                        parameters = param_list,
                        random = random,
                        map = list(),
                        hessian = TRUE,
                        DLL = "childSurvLL_TMBExports")
  
  opt <- nlminb(obj$par, obj$fn, obj$gr,
                lower = c(-15, -15, -15, -20, -20, -20, -20, -20, -20), #, -2),
                upper = c(5, 5, -2, 20, 20, 20, 20, 20, 20)) #, 4))
  SD0 <- TMB::sdreport(obj,
                       getJointPrecision = TRUE,
                       getReportCovariance = TRUE,
                       bias.correct = TRUE,
                       bias.correct.control = list(sd = TRUE))
  
  
  # TMB post-processing ------------------------------------------------------
  
  # obtain point estimates (means) for fixed and random effects
  mu <- c(SD0$par.fixed, SD0$par.random)
  
  # get ids for different parameters
  intercept.B0.idx <- which(names(mu) == "intercept_logit_B0")
  intercept.B1.idx <- which(names(mu) == "intercept_logit_B1")
  intercept.B2.idx <- which(names(mu) == "intercept_logit_B2")
  time.struct.B0.idx <- which(names(mu) == "delta_logit_B0")
  time.struct.B1.idx <- which(names(mu) == "delta_logit_B1")
  time.struct.B2.idx <- which(names(mu) == "delta_logit_B2")
  time.unstruct.B0.idx <- which(names(mu) == "epsilon_logit_B0")
  time.unstruct.B1.idx <- which(names(mu) == "epsilon_logit_B1")
  time.unstruct.B2.idx <- which(names(mu) == "epsilon_logit_B2")
  
  if (time_model == "spline") {
    
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
                                constrain.idx.list = list(time.struct.B0.idx,
                                                          time.struct.B1.idx,
                                                          time.struct.B2.idx),
                                A.mat.list = A.mat.list)
    
    # take the constrained draws
    t.draws <- t.draws$x.c
    
    # combine draws for linear predictor (B0)
    fitted_B0 <-
      matrix(rep(t.draws[intercept.B0.idx,], n_betas), nrow = n_betas, byrow = T) +
      t.draws[time.struct.B0.idx,]
    fitted_B0 <- B %*% fitted_B0
    
    # combine draws for linear predictor (B1)
    fitted_B1 <-
      matrix(rep(t.draws[intercept.B1.idx,], n_betas), nrow = n_betas, byrow = T) +
      t.draws[time.struct.B1.idx,]
    fitted_B1 <- B %*% fitted_B1
    
    # combine draws for linear predictor (B2)
    fitted_B2 <-
      matrix(rep(t.draws[intercept.B2.idx,], n_betas), nrow = n_betas, byrow = T) +
      t.draws[time.struct.B2.idx,]
    fitted_B2 <- B %*% fitted_B2
    
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
                                constrain.idx.list = list(time.struct.B0.idx,
                                                          time.struct.B1.idx,
                                                          time.struct.B2.idx),
                                A.mat.list = A.mat.list)
    
    # take the constrained draws
    t.draws <- t.draws$x.c
    
    # combine draws for linear predictor (logit B0)
    fitted_B0 <-
      matrix(rep(t.draws[intercept.B0.idx,], n_years), nrow = n_years, byrow = T) +
      t.draws[time.struct.B0.idx,]
    
    # combine draws for linear predictor (logit B1)
    fitted_B1 <-
      matrix(rep(t.draws[intercept.B1.idx,], n_years), nrow = n_years, byrow = T) +
      t.draws[time.struct.B1.idx,]
    
    # combine draws for linear predictor (logit B2)
    fitted_B2 <-
      matrix(rep(t.draws[intercept.B2.idx,], n_years), nrow = n_years, byrow = T) +
      t.draws[time.struct.B2.idx,]
    
    # add IID
    if (include_iid_in_pred) {
      fitted_B0 <- fitted_B0 + t.draws[time.unstruct.B0.idx,]
      fitted_B1 <- fitted_B1 + t.draws[time.unstruct.B1.idx,]
      fitted_B2 <- fitted_B2 + t.draws[time.unstruct.B2.idx,]
    }
    
  } else {
    stop("Invalid temporal model.")
  }
  
  # combine to get nmr, imr, u5mr
  nmr_mat <- matrix(nrow = n_years, ncol = 1000)
  imr_mat <- matrix(nrow = n_years, ncol = 1000)
  u5mr_mat <- matrix(nrow = n_years, ncol = 1000)
  
  for (t in 1:n_years) {
    nmr_mat[t,] <- 1-exp(-1 * expit(fitted_B0)[t,])
    imr_mat[t,] <- 1-exp(-1 * expit(fitted_B0)[t,])*exp(-11*expit(fitted_B1)[t,])
    u5mr_mat[t,] <- 1-exp(-1 * expit(fitted_B0)[t,])*exp(-11*expit(fitted_B1)[t,])*exp(-48*expit(fitted_B2)[t,])
  }
  
  #  create prediction data.frame
  df_pred <- data.frame(year = start_year:end_year)
  
  # get summaries
  df_pred$logit_B0_smoothed_med <- apply(fitted_B0, 1, quantile, 0.5)
  df_pred$logit_B0_smoothed_lower <- apply(fitted_B0, 1, quantile, 0.05)
  df_pred$logit_B0_smoothed_upper <- apply(fitted_B0, 1, quantile, 0.95)
  df_pred$logit_B1_smoothed_med <- apply(fitted_B1, 1, quantile, 0.5)
  df_pred$logit_B1_smoothed_lower <- apply(fitted_B1, 1, quantile, 0.05)
  df_pred$logit_B1_smoothed_upper <- apply(fitted_B1, 1, quantile, 0.95)
  df_pred$logit_B2_smoothed_med <- apply(fitted_B2, 1, quantile, 0.5)
  df_pred$logit_B2_smoothed_lower <- apply(fitted_B2, 1, quantile, 0.05)
  df_pred$logit_B2_smoothed_upper <- apply(fitted_B2, 1, quantile, 0.95)
  df_pred$nmr_smoothed_med <- apply(nmr_mat, 1, quantile, 0.5)
  df_pred$nmr_smoothed_lower <- apply(nmr_mat, 1, quantile, 0.05)
  df_pred$nmr_smoothed_upper <- apply(nmr_mat, 1, quantile, 0.95)
  df_pred$imr_smoothed_med <- apply(imr_mat, 1, quantile, 0.5)
  df_pred$imr_smoothed_lower <- apply(imr_mat, 1, quantile, 0.05)
  df_pred$imr_smoothed_upper <- apply(imr_mat, 1, quantile, 0.95)
  df_pred$u5mr_smoothed_med <- apply(u5mr_mat, 1, quantile, 0.5)
  df_pred$u5mr_smoothed_lower <- apply(u5mr_mat, 1, quantile, 0.05)
  df_pred$u5mr_smoothed_upper <- apply(u5mr_mat, 1, quantile, 0.95)
  
  return(df_pred)
}