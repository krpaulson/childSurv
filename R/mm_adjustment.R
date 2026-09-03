
#' Apply missing mothers adjustment
#' 
#' @description
#' Apply missing mothers adjustment to pssst output for piecewise-exponential
#' or log-logistic survival models. 
#'
#' @param fit_pssst fit object from the output of pssst::surv_synthetic()
#' @param ratios data.frame with columns 'year' and 'ratio' indicating the
#'   ratios to apply, equal to adjusted to unadjusted U5MR.
#' @param survival_model either 'piecewise_exponential' or 'log_logistic'
#' @param sy survey year
#'
#' @returns modified fit_pssst with new U5MR, IMR, NMR, and survival parameters.
#' @export
mm_adjustment <- function(fit_pssst, ratios, survival_model, sy) {
  
  # formatting
  df <- fit_pssst$result
  df$year <- (sy - 20):(sy - 1)
  df <- merge(df, ratios, by = "year", all.x = TRUE)
  if (any(is.na(df$ratio))) {
    df[is.na(df$ratio),]$ratio <- 1
  }

  # piecewise exponential ---------------------------------------------------
  if (survival_model == "piecewise_exponential") {
    
    # theta_t = [log(alpha1_t) log(alpha2_t) log(alpha3_t)]
    # theta_t = log(b_t) + theta*_t
    # b_t = log(1-r_t(1-S*_t(60))) / log(S*_t(60))
    
    if (mean(fit_pssst$result$`log_scale_mean_[0,1]`) > 0) {
      warning("Please double check that you have already converted log(scales) to log(rates).")
    }
    
    # compute b and apply to unadjusted parameters
    df$b <- log(1 - df$ratio*(df$U5MR)) / log(1 - df$U5MR)
    df$`log_scale_mean_[0,1]` <- log(df$b) + df$`log_scale_mean_[0,1]`
    df$`log_scale_mean_[1,12]` <- log(df$b) + df$`log_scale_mean_[1,12]`
    df$`log_scale_mean_[12,Inf]` <- log(df$b) + df$`log_scale_mean_[12,Inf]`
    
    # compute adjusted mortality rates from adjusted parameters
    df$U5MR <- 1 - exp(-1 * (60*exp(df$`log_scale_mean_[0,1]`) + 12*exp(df$`log_scale_mean_[1,12]`) + exp(df$`log_scale_mean_[12,Inf]`)))
    df$IMR <- 1 - exp(-1 * (12*exp(df$`log_scale_mean_[0,1]`) + 12*exp(df$`log_scale_mean_[1,12]`) + exp(df$`log_scale_mean_[12,Inf]`)))
    df$NMR <- 1 - exp(-1 * (exp(df$`log_scale_mean_[0,1]`) + exp(df$`log_scale_mean_[1,12]`) + exp(df$`log_scale_mean_[12,Inf]`)))
  
  # log-logistic --------------------------------------------------- 
  } else if (survival_model == "log_logistic") {
    
    # theta_t = [log(alpha1_t) log(alpha2_t) log(alpha3_t)]
    # theta_t = log(b_t) + theta*_t
    # b_t = log(1-r_t(1-S*_t(60))) / log(S*_t(60))
    
    # compute b and apply to unadjusted parameters
    # sigma = 1/shape; mu = scale
    df$b <- 60 / ((1/(1 - df$ratio*df$U5MR) - 1)^(1/plogis(df$log_shape_mean)) * exp(df$log_scale_mean))
    df$log_scale_mean <- log(df$b) + df$log_scale_mean
    
    # compute adjusted mortality rates from adjusted parameters
    df$U5MR <- flexsurv::pllogis(60, plogis(df$log_shape_mean), exp(df$log_scale_mean))
    df$IMR <- flexsurv::pllogis(12, plogis(df$log_shape_mean), exp(df$log_scale_mean))
    df$NMR <- flexsurv::pllogis(1, plogis(df$log_shape_mean), exp(df$log_scale_mean))
    
  } else {
    stop("Only 'piecewise_exponential' and 'log_logistic' survival models supported.")
  }
  
  fit_pssst$result <- df
  return(fit_pssst)
  
}

