#undef TMB_OBJECTIVE_PTR
#define TMB_OBJECTIVE_PTR obj

#include "childSurvLL/helpers.hpp"

template <class Type>
Type pe_vr_pspline_pois_lognormal(objective_function<Type>* obj) {

  using namespace density;
  using namespace Eigen;
  
  /////////////////////
  // get things from R
  /////////////////////
  
  // data
  DATA_MATRIX(B); // B-spline matrix
  DATA_MATRIX(A); // Constraint matrix
  DATA_SPARSE_MATRIX(R); // Structure matrix for random walk on spline coefficients
  DATA_INTEGER(n_obs_vr);
  DATA_IVECTOR(time_id_vr);
  DATA_VECTOR(months_vr);
  DATA_VECTOR(obs_vr);
  DATA_VECTOR(n_vr);
  DATA_VECTOR(births);
  DATA_VECTOR(pop);
  DATA_INTEGER(n_obs_direct);
  DATA_IVECTOR(time_id_direct);
  DATA_VECTOR(months_direct);
  DATA_VECTOR(obs_direct);
  DATA_VECTOR(se_direct);
  int n_betas = B.cols();
  
  // parameters
  PARAMETER(intercept_log_alpha0);
  PARAMETER(intercept_log_alpha1);
  PARAMETER(intercept_log_alpha2);
  PARAMETER(log_tau_delta_log_alpha0);
  PARAMETER(log_tau_delta_log_alpha1);
  PARAMETER(log_tau_delta_log_alpha2);
  PARAMETER(log_tau_epsilon);
  PARAMETER_VECTOR(delta_log_alpha0);
  PARAMETER_VECTOR(delta_log_alpha1);
  PARAMETER_VECTOR(delta_log_alpha2);
  PARAMETER_VECTOR(epsilon);
  
  ////////////////////////
  // priors + hyperpriors
  ////////////////////////
  
  // initialize
  Type nll = 0.0;
  
  // intercepts
  nll -= dnorm(intercept_log_alpha0, Type(0.0), Type(30.00), true);
  nll -= dnorm(intercept_log_alpha1, Type(0.0), Type(30.00), true);
  nll -= dnorm(intercept_log_alpha2, Type(0.0), Type(30.00), true);
  
  // RW2 (alpha0)
  Eigen::SparseMatrix<Type> Q0(n_betas, n_betas);
  for (int i = 0; i < n_betas; i++) {
    for (int j = 0; j < n_betas; j++) {
      Q0.coeffRef(i, j) = R.coeffRef(i, j) * exp(log_tau_delta_log_alpha0);
    }
  }
  nll += GMRF(Q0)(delta_log_alpha0);
  
  // RW2 (alpha1)
  Eigen::SparseMatrix<Type> Q1(n_betas, n_betas);
  for (int i = 0; i < n_betas; i++) {
    for (int j = 0; j < n_betas; j++) {
      Q1.coeffRef(i, j) = R.coeffRef(i, j) * exp(log_tau_delta_log_alpha1);
    }
  }
  nll += GMRF(Q1)(delta_log_alpha1);
  
  // RW2 (alpha2)
  Eigen::SparseMatrix<Type> Q2(n_betas, n_betas);
  for (int i = 0; i < n_betas; i++) {
    for (int j = 0; j < n_betas; j++) {
      Q2.coeffRef(i, j) = R.coeffRef(i, j) * exp(log_tau_delta_log_alpha2);
    }
  }
  nll += GMRF(Q2)(delta_log_alpha2);
  
  // hyperpriors
  nll -= dpcprec(log_tau_delta_log_alpha0, Type(0.001), Type(0.5), true);
  nll -= dpcprec(log_tau_delta_log_alpha1, Type(0.001), Type(0.5), true);
  nll -= dpcprec(log_tau_delta_log_alpha2, Type(0.001), Type(0.5), true);
  //nll -= dlgamma(log_tau_delta_log_alpha0, Type(1.0), Type(1/0.00005), true);
  //nll -= dlgamma(log_tau_delta_log_alpha1, Type(1.0), Type(1/0.00005), true);
  //nll -= dlgamma(log_tau_delta_log_alpha2, Type(1.0), Type(1/0.00005), true);
  //nll -= dnorm(log_tau_delta_log_alpha0, Type(2.0), Type(0.001), true);
  //nll -= dnorm(log_tau_delta_log_alpha1, Type(2.0), Type(0.001), true);
  //nll -= dnorm(log_tau_delta_log_alpha2, Type(2.0), Type(0.001), true);
  
  // overdispersion
  Type sd_epsilon = exp(-0.5 * log_tau_epsilon);
  for (int i = 0; i < n_obs_vr; i++) {
    nll -= dnorm(epsilon(i), Type(0.0), sd_epsilon, true);
  }
  nll -= dpcprec(log_tau_epsilon, Type(0.1), Type(0.5), true);
  
  
  ////////////////
  // constraints
  ////////////////
  
  // a constrained vector x_c for an unconstrained vector x is calculated as:
  // x_c = x - Q^{-1}A'(AQ^{-1}A')^{-1}(Ax-e)
  // for constraint Ax=e, and for GMRF x with precision Q
  
  // Invert precision matrices (possible since we added a small value to the diagonal)
  matrix<Type> Q_inv_rw2_log_alpha0 = invertSparseMatrix(Q0);
  matrix<Type> Q_inv_rw2_log_alpha1 = invertSparseMatrix(Q1);
  matrix<Type> Q_inv_rw2_log_alpha2 = invertSparseMatrix(Q2);
  
  // Create A^T
  matrix<Type> A_rw2_T_log_alpha0 = A.transpose();
  matrix<Type> A_rw2_T_log_alpha1 = A.transpose();
  matrix<Type> A_rw2_T_log_alpha2 = A.transpose();
  
  // Create Q^{-1}A^T
  matrix<Type> QinvA_rw2_log_alpha0 = Q_inv_rw2_log_alpha0 * A_rw2_T_log_alpha0;
  matrix<Type> QinvA_rw2_log_alpha1 = Q_inv_rw2_log_alpha1 * A_rw2_T_log_alpha1;
  matrix<Type> QinvA_rw2_log_alpha2 = Q_inv_rw2_log_alpha2 * A_rw2_T_log_alpha2;
  
  // Create AQ^{-1}A^T
  matrix<Type> AQinvA_rw2_log_alpha0 = A * QinvA_rw2_log_alpha0;
  matrix<Type> AQinvA_rw2_log_alpha1 = A * QinvA_rw2_log_alpha1;
  matrix<Type> AQinvA_rw2_log_alpha2 = A * QinvA_rw2_log_alpha2;
  
  // Create (AQ^{-1}A^T)^{-1}
  matrix<Type> AQinvA_rw2_inv_log_alpha0 = AQinvA_rw2_log_alpha0.inverse(); // okay for small matrices
  matrix<Type> AQinvA_rw2_inv_log_alpha1 = AQinvA_rw2_log_alpha1.inverse();
  matrix<Type> AQinvA_rw2_inv_log_alpha2 = AQinvA_rw2_log_alpha2.inverse();
  
  // Create Ax
  matrix<Type> Ax_rw2_log_alpha0 = (A * delta_log_alpha0.matrix());
  matrix<Type> Ax_rw2_log_alpha1 = (A * delta_log_alpha1.matrix());
  matrix<Type> Ax_rw2_log_alpha2 = (A * delta_log_alpha2.matrix());
  
  // Convert Q^{-1}A'(AQ^{-1}A')^{-1}(Ax-e) to vector form for conditioning by kriging correction
  matrix<Type> krig_correct_log_alpha0 = QinvA_rw2_log_alpha0 * AQinvA_rw2_inv_log_alpha0 * Ax_rw2_log_alpha0;
  matrix<Type> krig_correct_log_alpha1 = QinvA_rw2_log_alpha1 * AQinvA_rw2_inv_log_alpha1 * Ax_rw2_log_alpha1;
  matrix<Type> krig_correct_log_alpha2 = QinvA_rw2_log_alpha2 * AQinvA_rw2_inv_log_alpha2 * Ax_rw2_log_alpha2;
  vector<Type> krig_correct_vec_log_alpha0(n_betas);
  vector<Type> krig_correct_vec_log_alpha1(n_betas);
  vector<Type> krig_correct_vec_log_alpha2(n_betas);
  for (int i = 0; i < n_betas; i++) {
    krig_correct_vec_log_alpha0(i) = krig_correct_log_alpha0(i,0);
    krig_correct_vec_log_alpha1(i) = krig_correct_log_alpha1(i,0);
    krig_correct_vec_log_alpha2(i) = krig_correct_log_alpha2(i,0);
  }
  
  // Construct constrained vector x_c = x - Q^{-1}A'(AQ^{-1}A')^{-1}(Ax-e)
  vector<Type> delta_log_alpha0_c = delta_log_alpha0 - krig_correct_vec_log_alpha0;
  vector<Type> delta_log_alpha1_c = delta_log_alpha1 - krig_correct_vec_log_alpha1;
  vector<Type> delta_log_alpha2_c = delta_log_alpha2 - krig_correct_vec_log_alpha2;
  
  ////////////////
  // likelihood
  ////////////////
  
  // beta_k = intercept + delta_k
  vector<Type> beta_log_alpha0(n_betas);
  vector<Type> beta_log_alpha1(n_betas);
  vector<Type> beta_log_alpha2(n_betas);
  for (int k = 0; k < n_betas; k++) {
    beta_log_alpha0(k) =  intercept_log_alpha0 + delta_log_alpha0_c(k);
    beta_log_alpha1(k) = intercept_log_alpha1 + delta_log_alpha1_c(k);
    beta_log_alpha2(k) = intercept_log_alpha2 + delta_log_alpha2_c(k);
  }
  
  Type log_alpha0;
  Type log_alpha1;
  Type log_alpha2;
  Type alpha0;
  Type alpha1;
  Type alpha2;
  Type lambda;
  Type mx;
  for (int i=0; i<n_obs_vr; i++) {
    int idx_time = time_id_vr(i) - 1;
    vector<Type> B_t = B.row(idx_time);
    log_alpha0 = (B_t*beta_log_alpha0).sum();
    log_alpha1 = (B_t*beta_log_alpha1).sum();
    log_alpha2 = (B_t*beta_log_alpha2).sum();
    alpha0 = exp(log_alpha0);
    alpha1 = exp(log_alpha1);
    alpha2 = exp(log_alpha2);
    // nn
    if (months_vr[i] == 0 && n_vr[i] == 1) {
      lambda = (1 - exp(-1 * (alpha0 + alpha1 + alpha2))) * exp(epsilon[i]);
      nll -= dbinom(obs_vr[i], births[i], lambda, true);
    // pnn
    } else if (months_vr[i] == 1 && n_vr[i] == 11) {
      mx = 12 * (1 - exp(-1 * (12*alpha0 + 12*alpha1 + alpha2))) /
        ((1 - exp(-1 * (alpha0 + alpha1 + alpha2))) / (alpha0 + alpha1 + alpha2) +
          ((exp(-1 * (alpha0 + alpha1 + alpha2)) - exp(-1 * (12*alpha0 + 12*alpha1 + alpha2))) / (alpha0 + alpha1)));
      lambda = pop[i] * mx - births[i] * (1 - exp(-1 * (alpha0 + alpha1 + alpha2)));
      lambda = lambda * exp(epsilon[i]);
      if (lambda < 0) {
        lambda = 0;
      }
      nll -= dpois(obs_vr[i], lambda, true);
    // infant
    } else if (months_vr[i] == 0 && n_vr[i] == 12) {
      mx = 12 * (1 - exp(-1 * (12*alpha0 + 12*alpha1 + alpha2))) /
        ((1 - exp(-1 * (alpha0 + alpha1 + alpha2))) / (alpha0 + alpha1 + alpha2) +
          ((exp(-1 * (alpha0 + alpha1 + alpha2)) - exp(-1 * (12*alpha0 + 12*alpha1 + alpha2))) / (alpha0 + alpha1)));
      lambda = pop[i] * mx * exp(epsilon[i]);
      nll -= dpois(obs_vr[i], lambda, true);
    // 1-4 years
    } else {
      mx = 12 * alpha0;
      lambda = pop[i] * mx * exp(epsilon[i]);
      nll -= dpois(obs_vr[i], lambda, true);
    }
  }
  
  // direct observations of logit mortality rates
  if (n_obs_direct > 0) {
    Type xq0;
    Type logit_xq0;
    for (int i=0; i<n_obs_direct; i++) {
      int idx_time = time_id_direct(i) - 1;
      vector<Type> B_t = B.row(idx_time);
      log_alpha0 = (B_t*beta_log_alpha0).sum();
      log_alpha1 = (B_t*beta_log_alpha1).sum();
      log_alpha2 = (B_t*beta_log_alpha2).sum();
      if (months_direct[i] == 1) {
        xq0 = 1 - exp(-1 * (exp(log_alpha0) + exp(log_alpha1) + exp(log_alpha2)));
      } else if (months_direct[i] == 12) {
        xq0 = 1 - exp(-1 * (12*exp(log_alpha0) + 12*exp(log_alpha1) + exp(log_alpha2)));
      } else if (months_direct[i] == 60) {
        xq0 = 1 - exp(-1 * (60*exp(log_alpha0) + 12*exp(log_alpha1) + exp(log_alpha2)));
      }
      logit_xq0 = log(xq0 / (1-xq0));
      nll -= dnorm(obs_direct(i), logit_xq0, se_direct(i), true);
    }
  }

  return nll;
}

#undef TMB_OBJECTIVE_PTR
#define TMB_OBJECTIVE_PTR this