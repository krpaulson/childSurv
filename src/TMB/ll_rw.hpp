#undef TMB_OBJECTIVE_PTR
#define TMB_OBJECTIVE_PTR obj

#include "childSurv/helpers.hpp"

template <class Type>
Type ll_rw(objective_function<Type>* obj) {

  using namespace density;
  using namespace Eigen;
  
  /////////////////////
  // get things from R
  /////////////////////
  
  // data (general)
  DATA_INTEGER(n_years); // Number of consecutive years to make estimates for
  DATA_SPARSE_MATRIX(R); // Structure matrix for random walk
  
  // data (pseudo-likelihood estimates)
  DATA_INTEGER(n_obs_fbh); // Number of pseud-likelihood estimates;
  DATA_VECTOR(thetahat); // Pseudo-likelihood estimates
  DATA_VECTOR(V); // Variance-covariance matrix for pseudo-likelihood estimates
  DATA_IVECTOR(time_id_fbh); // Numeric indicator for which time point an observation belongs to
  DATA_IVECTOR(par_id_fbh); // Numeric indicator for which parameter an observation belongs to (1=shape; 2=scale)
  
  // data (VR)
  DATA_INTEGER(n_obs_vr); // Number of VR observations
  DATA_IVECTOR(time_id_vr); // Numeric indicator for which time point a VR observation belongs to
  DATA_VECTOR(months_vr); // Data are for nDx -- this input tells us what x is in months
  DATA_VECTOR(n_vr); // Data are for nDx -- this input tells us what n is in months
  DATA_VECTOR(obs_vr); // Observations of nDx (death counts)
  DATA_VECTOR(births); // Births in year of observation
  DATA_VECTOR(pop); // Mid-year population in age-group and year of observation
  
  // data (Pre-processed)
  DATA_INTEGER(n_obs_pp); // Number of pre-processed estimates
  DATA_IVECTOR(time_id_pp); // Numeric indicator for which time point a pre-processed estimate belongs to
  DATA_VECTOR(months_pp); // Pre-processed estimates are for logit(nq0) -- this input tells us what n is
  DATA_VECTOR(obs_pp); // Pre-processed estimate of logit(nq0)
  DATA_VECTOR(se_pp); // Standard error of pre-processed estimate
  
  // parameters
  PARAMETER(intercept_logit_shape);
  PARAMETER(intercept_log_scale);
  PARAMETER(log_tau_delta_logit_shape); // precision for RW2 terms
  PARAMETER(log_tau_delta_log_scale);
  PARAMETER(log_tau_epsilon_logit_shape); // precision for IID terms
  PARAMETER(log_tau_epsilon_log_scale);
  PARAMETER_VECTOR(delta_logit_shape); // RW2 terms
  PARAMETER_VECTOR(delta_log_scale);
  PARAMETER_VECTOR(epsilon_logit_shape); // IID terms
  PARAMETER_VECTOR(epsilon_log_scale);
  
  ////////////////////////
  // priors + hyperpriors
  ////////////////////////
  
  // initialize
  Type nll = 0.0;
  
  // intercepts
  nll -= dnorm(intercept_logit_shape, Type(-1.2), Type(1.4), true);
  nll -= dnorm(intercept_log_scale, Type(23), Type(35), true);
  
  // RW2 (shape)
  Eigen::SparseMatrix<Type> Q_shape(n_years, n_years);
  for (int i = 0; i < n_years; i++) {
    for (int j = 0; j < n_years; j++) {
      Q_shape.coeffRef(i, j) = R.coeffRef(i, j) * exp(log_tau_delta_logit_shape);
    }
  }
  nll += GMRF(Q_shape)(delta_logit_shape);
  
  // RW2 (scale)
  Eigen::SparseMatrix<Type> Q_scale(n_years, n_years);
  for (int i = 0; i < n_years; i++) {
    for (int j = 0; j < n_years; j++) {
      Q_scale.coeffRef(i, j) = R.coeffRef(i, j) * exp(log_tau_delta_log_scale);
    }
  }
  nll += GMRF(Q_scale)(delta_log_scale);
  
  // iid effects
  Type sd_epsilon_logit_shape = exp(-0.5 * log_tau_epsilon_logit_shape);
  Type sd_epsilon_log_scale = exp(-0.5 * log_tau_epsilon_log_scale);
  for (int i = 0; i < n_years; i++) {
    nll -= dnorm(epsilon_logit_shape(i), Type(0.0), sd_epsilon_logit_shape, true);
    nll -= dnorm(epsilon_log_scale(i), Type(0.0), sd_epsilon_log_scale, true);
  }
  
  // hyperpriors
  nll -= dpcprec(log_tau_delta_logit_shape, Type(10.0), Type(0.5), true); //1.0, 0.01
  nll -= dpcprec(log_tau_delta_log_scale, Type(10.0), Type(0.5), true); //1.0, 0.01
  //nll -= dlgamma(log_tau_delta_logit_shape, Type(1.0), Type(1/0.00005), true);
  //nll -= dlgamma(log_tau_delta_log_scale, Type(1.0), Type(1/0.00005), true);
  nll -= dlgamma(log_tau_epsilon_logit_shape, Type(1.0), Type(1/0.00005), true);
  nll -= dlgamma(log_tau_epsilon_log_scale, Type(1.0), Type(1/0.00005), true);
  
  ////////////////
  // constraints
  ////////////////
  
  // a constrained vector x_c for an unconstrained vector x is calculated as:
  // x_c = x - Q^{-1}A'(AQ^{-1}A')^{-1}(Ax-e)
  // for constraint Ax=e, and for GMRF x with precision Q
  
  // Invert precision matrices (possible since we added a small value to the diagonal)
  matrix<Type> Q_inv_rw2_shape = invertSparseMatrix(Q_shape);
  matrix<Type> Q_inv_rw2_scale = invertSparseMatrix(Q_scale);
  
  // Create constraint matrices A
  matrix<Type> A_rw2_shape(1, n_years);
  matrix<Type> A_rw2_scale(1, n_years);
  for(int i = 0; i < n_years; i++) {
    A_rw2_shape(0, i) = 1; // sum-to-0 constraint
    A_rw2_scale(0, i) = 1;
  }
  
  // Create A^T
  matrix<Type> A_rw2_T_shape = A_rw2_shape.transpose();
  matrix<Type> A_rw2_T_scale = A_rw2_scale.transpose();
  
  // Create Q^{-1}A^T
  matrix<Type> QinvA_rw2_shape = Q_inv_rw2_shape * A_rw2_T_shape;
  matrix<Type> QinvA_rw2_scale = Q_inv_rw2_scale * A_rw2_T_scale;
  
  // Create AQ^{-1}A^T
  matrix<Type> AQinvA_rw2_shape = A_rw2_shape * QinvA_rw2_shape;
  matrix<Type> AQinvA_rw2_scale = A_rw2_scale * QinvA_rw2_scale;
  
  // Create (AQ^{-1}A^T)^{-1}
  matrix<Type> AQinvA_rw2_inv_shape = AQinvA_rw2_shape.inverse(); // okay for small matrices
  matrix<Type> AQinvA_rw2_inv_scale = AQinvA_rw2_scale.inverse();
  
  // Create Ax
  matrix<Type> Ax_rw2_shape = (A_rw2_shape * delta_logit_shape.matrix());
  matrix<Type> Ax_rw2_scale = (A_rw2_scale * delta_log_scale.matrix());
  
  // Convert Ax from matrix to vector form - needed for dnorm & MVNORM
  vector<Type> Ax_rw2_vec_shape(1);
  vector<Type> Ax_rw2_vec_scale(1);
  for(int i = 0; i < 1; i++) {
    Ax_rw2_vec_shape(i) = Ax_rw2_shape(i,0);
    Ax_rw2_vec_scale(i) = Ax_rw2_scale(i,0);
  }
  
  // Convert Q^{-1}A'(AQ^{-1}A')^{-1}(Ax-e) to vector form for conditioning by kriging correction
  matrix<Type> krig_correct_shape = QinvA_rw2_shape * AQinvA_rw2_inv_shape * Ax_rw2_shape;
  matrix<Type> krig_correct_scale = QinvA_rw2_scale * AQinvA_rw2_inv_scale * Ax_rw2_scale;
  vector<Type> krig_correct_vec_shape(n_years);
  vector<Type> krig_correct_vec_scale(n_years);
  for (int i = 0; i < n_years; i++) {
    krig_correct_vec_shape(i) = krig_correct_shape(i,0);
    krig_correct_vec_scale(i) = krig_correct_scale(i,0);
  }
  
  // Construct constrained vector x_c = x - Q^{-1}A'(AQ^{-1}A')^{-1}(Ax-e)
  vector<Type> delta_logit_shape_c = delta_logit_shape - krig_correct_vec_shape;
  vector<Type> delta_log_scale_c = delta_log_scale - krig_correct_vec_scale;
  
  ////////////////
  // likelihood
  ////////////////
  
  // pseudo-likelihood estimates from FBH
  vector<Type> x(n_obs_fbh);
  Type theta;
  for(int i = 0; i < n_obs_fbh; i++){
    int idx_time = time_id_fbh(i) - 1; // fetch time index for this observation (0-indexing)
    if(par_id_fbh(i) == 1) {
      theta = intercept_logit_shape + delta_logit_shape_c(idx_time) + epsilon_logit_shape(idx_time);
    }
    if(par_id_fbh(i) == 2) {
      theta = intercept_log_scale + delta_log_scale_c(idx_time) + epsilon_log_scale(idx_time);
    }
    x(i) = thetahat(i) - theta; // build observation vector, thetahat - theta
  }
  // likelihood (thetahat-theta ~ N(0,V))
  if (n_obs_fbh > 0) {
    MVNORM_t<Type> neg_log_dmvnorm(V);
    nll += neg_log_dmvnorm(x); 
  }
  
  // VR
  //  group 1: nn
  //  group 2: pnn if nn and <1 present
  //  group 3: <1 if nn not present
  //  group 4: 1, 2, 3, 4, or 1-4
  Type logit_shape;
  Type log_scale;
  Type lambda;
  Type mx;
  for (int i=0; i<n_obs_vr; i++) {
    int idx_time = time_id_vr(i) - 1;
    logit_shape = intercept_logit_shape + delta_logit_shape_c(idx_time) + epsilon_logit_shape(idx_time);
    log_scale = intercept_log_scale + delta_log_scale_c(idx_time) + epsilon_log_scale(idx_time);
    survfunc<Type> s(logit_shape, log_scale);
    // nn
    if (months_vr[i] == 0 && n_vr[i] == 1) {
      lambda = 1 - s(Type(1.0));
      nll -= dbinom(obs_vr[i], births[i], lambda, true);
    // pnn
    } else if (months_vr[i] == 1 && n_vr[i] == 11) {
      mx = 12 * (1 - s(Type(12.0))) / romberg::integrate(s, Type(0.0), Type(12.0));
      lambda = (pop[i] * mx - births[i] * (1 - s(Type(1.0))));
      if (lambda < 0) {
        lambda = 0;
      }
      nll -= dpois(obs_vr[i], lambda, true);
    // otherwise
    } else {
      mx = 12 * (s(months_vr[i]) - s(months_vr[i]+n_vr[i])) /
        romberg::integrate(s, months_vr[i], months_vr[i] + n_vr[i]);
      lambda = (pop[i] * mx);
      nll -= dpois(obs_vr[i], lambda, true);
    }
  }
  
  // pre-processed observations of logit mortality rates
  Type logit_xq0;
  for (int i=0; i<n_obs_pp; i++) {
    int idx_time = time_id_pp(i) - 1;
    logit_shape = intercept_logit_shape + delta_logit_shape_c(idx_time) + epsilon_logit_shape(idx_time);
    log_scale = intercept_log_scale + delta_log_scale_c(idx_time) + epsilon_log_scale(idx_time);
    logit_xq0 = (exp(logit_shape)/(1+exp(logit_shape))) * (log(months_pp(i)) - log_scale);
    nll -= dnorm(obs_pp(i), logit_xq0, se_pp(i), true);
  }
  
  return nll;
}

#undef TMB_OBJECTIVE_PTR
#define TMB_OBJECTIVE_PTR this