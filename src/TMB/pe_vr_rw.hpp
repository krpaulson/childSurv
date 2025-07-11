#undef TMB_OBJECTIVE_PTR
#define TMB_OBJECTIVE_PTR obj

#include "childSurv/helpers.hpp"

template <class Type>
Type pe_vr_rw(objective_function<Type>* obj) {

  using namespace density;
  using namespace Eigen;
  
  /////////////////////
  // get things from R
  /////////////////////
  
  // data
  DATA_INTEGER(n_years); // number of consecutive years to make estimates for
  DATA_SPARSE_MATRIX(R); // Structure matrix for random walk
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
  
  // parameters
  PARAMETER(intercept_log_alpha0);
  PARAMETER(intercept_log_alpha1);
  PARAMETER(intercept_log_alpha2);
  PARAMETER(log_tau_delta_log_alpha0);
  PARAMETER(log_tau_delta_log_alpha1);
  PARAMETER(log_tau_delta_log_alpha2);
  PARAMETER(log_tau_epsilon_log_alpha0);
  PARAMETER(log_tau_epsilon_log_alpha1);
  PARAMETER(log_tau_epsilon_log_alpha2);
  PARAMETER_VECTOR(delta_log_alpha0);
  PARAMETER_VECTOR(delta_log_alpha1);
  PARAMETER_VECTOR(delta_log_alpha2);
  PARAMETER_VECTOR(epsilon_log_alpha0);
  PARAMETER_VECTOR(epsilon_log_alpha1);
  PARAMETER_VECTOR(epsilon_log_alpha2);
  
  ////////////////////////
  // priors + hyperpriors
  ////////////////////////
  
  // initialize
  Type nll = 0.0;
  
  // intercepts
  nll -= dnorm(intercept_log_alpha0, Type(0.0), Type(30.00), true);
  nll -= dnorm(intercept_log_alpha1, Type(0.0), Type(30.00), true);
  nll -= dnorm(intercept_log_alpha2, Type(0.0), Type(30.00), true);
  
  // RW2 (Beta0)
  Eigen::SparseMatrix<Type> Q0(n_years, n_years);
  for (int i = 0; i < n_years; i++) {
    for (int j = 0; j < n_years; j++) {
      Q0.coeffRef(i, j) = R.coeffRef(i, j) * exp(log_tau_delta_log_alpha0);
    }
  }
  nll += GMRF(Q0)(delta_log_alpha0);
  
  // RW2 (Beta1)
  Eigen::SparseMatrix<Type> Q1(n_years, n_years);
  for (int i = 0; i < n_years; i++) {
    for (int j = 0; j < n_years; j++) {
      Q1.coeffRef(i, j) = R.coeffRef(i, j) * exp(log_tau_delta_log_alpha1);
    }
  }
  nll += GMRF(Q1)(delta_log_alpha1);
  
  // RW2 (Beta2)
  Eigen::SparseMatrix<Type> Q2(n_years, n_years);
  for (int i = 0; i < n_years; i++) {
    for (int j = 0; j < n_years; j++) {
      Q2.coeffRef(i, j) = R.coeffRef(i, j) * exp(log_tau_delta_log_alpha2);
    }
  }
  nll += GMRF(Q2)(delta_log_alpha2);
  
  // iid effects
  Type sd_epsilon_log_alpha0 = exp(-0.5 * log_tau_epsilon_log_alpha0);
  Type sd_epsilon_log_alpha1 = exp(-0.5 * log_tau_epsilon_log_alpha1);
  Type sd_epsilon_log_alpha2 = exp(-0.5 * log_tau_epsilon_log_alpha2);
  for (int i = 0; i < n_years; i++) {
    nll -= dnorm(epsilon_log_alpha0(i), Type(0.0), sd_epsilon_log_alpha0, true);
    nll -= dnorm(epsilon_log_alpha1(i), Type(0.0), sd_epsilon_log_alpha1, true);
    nll -= dnorm(epsilon_log_alpha2(i), Type(0.0), sd_epsilon_log_alpha2, true);
  }
  
  // hyperpriors
  //nll -= dpcprec(log_tau_delta_log_B0, Type(0.001), Type(0.5), true);
  //nll -= dpcprec(log_tau_delta_log_B1, Type(0.001), Type(0.5), true);
  //nll -= dpcprec(log_tau_delta_log_B2, Type(0.001), Type(0.5), true);
  nll -= dlgamma(log_tau_delta_log_alpha0, Type(1.0), Type(1/0.00005), true);
  nll -= dlgamma(log_tau_delta_log_alpha1, Type(1.0), Type(1/0.00005), true);
  nll -= dlgamma(log_tau_delta_log_alpha2, Type(1.0), Type(1/0.00005), true);
  nll -= dlgamma(log_tau_epsilon_log_alpha0, Type(1.0), Type(1/0.00005), true);
  nll -= dlgamma(log_tau_epsilon_log_alpha1, Type(1.0), Type(1/0.00005), true);
  nll -= dlgamma(log_tau_epsilon_log_alpha2, Type(1.0), Type(1/0.00005), true);
  
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
  
  // Create constraint matrices A
  matrix<Type> A_rw2_log_alpha0(1, n_years);
  matrix<Type> A_rw2_log_alpha1(1, n_years);
  matrix<Type> A_rw2_log_alpha2(1, n_years);
  for(int i = 0; i < n_years; i++) {
    A_rw2_log_alpha0(0, i) = 1; // sum-to-0 constraint
    A_rw2_log_alpha1(0, i) = 1;
    A_rw2_log_alpha2(0, i) = 1;
  }
  
  // Create A^T
  matrix<Type> A_rw2_T_log_alpha0 = A_rw2_log_alpha0.transpose();
  matrix<Type> A_rw2_T_log_alpha1 = A_rw2_log_alpha1.transpose();
  matrix<Type> A_rw2_T_log_alpha2 = A_rw2_log_alpha2.transpose();
  
  // Create Q^{-1}A^T
  matrix<Type> QinvA_rw2_log_alpha0 = Q_inv_rw2_log_alpha0 * A_rw2_T_log_alpha0;
  matrix<Type> QinvA_rw2_log_alpha1 = Q_inv_rw2_log_alpha1 * A_rw2_T_log_alpha1;
  matrix<Type> QinvA_rw2_log_alpha2 = Q_inv_rw2_log_alpha2 * A_rw2_T_log_alpha2;
  
  // Create AQ^{-1}A^T
  matrix<Type> AQinvA_rw2_log_alpha0 = A_rw2_log_alpha0 * QinvA_rw2_log_alpha0;
  matrix<Type> AQinvA_rw2_log_alpha1 = A_rw2_log_alpha1 * QinvA_rw2_log_alpha1;
  matrix<Type> AQinvA_rw2_log_alpha2 = A_rw2_log_alpha2 * QinvA_rw2_log_alpha2;
  
  // Create (AQ^{-1}A^T)^{-1}
  matrix<Type> AQinvA_rw2_inv_log_alpha0 = AQinvA_rw2_log_alpha0.inverse(); // okay for small matrices
  matrix<Type> AQinvA_rw2_inv_log_alpha1 = AQinvA_rw2_log_alpha1.inverse();
  matrix<Type> AQinvA_rw2_inv_log_alpha2 = AQinvA_rw2_log_alpha2.inverse();
  
  // Create Ax
  matrix<Type> Ax_rw2_log_alpha0 = (A_rw2_log_alpha0 * delta_log_alpha0.matrix());
  matrix<Type> Ax_rw2_log_alpha1 = (A_rw2_log_alpha1 * delta_log_alpha1.matrix());
  matrix<Type> Ax_rw2_log_alpha2 = (A_rw2_log_alpha2 * delta_log_alpha2.matrix());
  
  // Convert Ax from matrix to vector form - needed for dnorm & MVNORM
  vector<Type> Ax_rw2_vec_log_alpha0(1);
  vector<Type> Ax_rw2_vec_log_alpha1(1);
  vector<Type> Ax_rw2_vec_log_alpha2(1);
  for(int i = 0; i < 1; i++) {
    Ax_rw2_vec_log_alpha0(i) = Ax_rw2_log_alpha0(i,0);
    Ax_rw2_vec_log_alpha1(i) = Ax_rw2_log_alpha1(i,0);
    Ax_rw2_vec_log_alpha2(i) = Ax_rw2_log_alpha2(i,0);
  }
  
  // Convert Q^{-1}A'(AQ^{-1}A')^{-1}(Ax-e) to vector form for conditioning by kriging correction
  matrix<Type> krig_correct_log_alpha0 = QinvA_rw2_log_alpha0 * AQinvA_rw2_inv_log_alpha0 * Ax_rw2_log_alpha0;
  matrix<Type> krig_correct_log_alpha1 = QinvA_rw2_log_alpha1 * AQinvA_rw2_inv_log_alpha1 * Ax_rw2_log_alpha1;
  matrix<Type> krig_correct_log_alpha2 = QinvA_rw2_log_alpha2 * AQinvA_rw2_inv_log_alpha2 * Ax_rw2_log_alpha2;
  vector<Type> krig_correct_vec_log_alpha0(n_years);
  vector<Type> krig_correct_vec_log_alpha1(n_years);
  vector<Type> krig_correct_vec_log_alpha2(n_years);
  for (int i = 0; i < n_years; i++) {
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
  
  Type log_alpha0;
  Type log_alpha1;
  Type log_alpha2;
  Type alpha0;
  Type alpha1;
  Type alpha2;
  Type lambda;
  Type mx;
  Type prob;
  for (int i=0; i<n_obs_vr; i++) {
    int idx_time = time_id_vr(i) - 1;
    log_alpha0 = intercept_log_alpha0 + delta_log_alpha0_c(idx_time) + epsilon_log_alpha0(idx_time);
    log_alpha1 = intercept_log_alpha1 + delta_log_alpha1_c(idx_time) + epsilon_log_alpha1(idx_time);
    log_alpha2 = intercept_log_alpha2 + delta_log_alpha2_c(idx_time) + epsilon_log_alpha2(idx_time);
    alpha0 = exp(log_alpha0);
    alpha1 = exp(log_alpha1);
    alpha2 = exp(log_alpha2);
    // nn
    if (months_vr[i] == 0 && n_vr[i] == 1) {
      lambda = 1 - exp(-1 * (alpha0 + alpha1 + alpha2));
      nll -= dbinom(obs_vr[i], births[i], lambda, true);
    // pnn
    } else if (months_vr[i] == 1 && n_vr[i] == 11) {
      mx = 12 * (1 - exp(-1 * (12*alpha0 + 12*alpha1 + alpha2))) /
        ((1 - exp(-1 * (alpha0 + alpha1 + alpha2))) / (alpha0 + alpha1 + alpha2) +
          ((exp(-1 * (alpha0 + alpha1 + alpha2)) - exp(-1 * (12*alpha0 + 12*alpha1 + alpha2))) / (alpha0 + alpha1)));
      lambda = (pop[i] * mx - births[i] * 1 - exp(-1 * (alpha0 + alpha1 + alpha2)));
      if (lambda < 0) {
        lambda = 0;
    }
      nll -= dpois(obs_vr[i], lambda, true);
    // infant
    } else if (months_vr[i] == 0 && n_vr[i] == 12) {
      mx = 12 * (1 - exp(-1 * (12*alpha0 + 12*alpha1 + alpha2))) /
        ((1 - exp(-1 * (alpha0 + alpha1 + alpha2))) / (alpha0 + alpha1 + alpha2) +
          ((exp(-1 * (alpha0 + alpha1 + alpha2)) - exp(-1 * (12*alpha0 + 12*alpha1 + alpha2))) / (alpha0 + alpha1)));
      lambda = (pop[i] * mx);
      nll -= dpois(obs_vr[i], lambda, true);
    // 1-4 years
    } else {
      mx = 12 * alpha0;
      lambda = (pop[i] * mx);
      nll -= dpois(obs_vr[i], lambda, true);
    }
  }
  
  // direct observations of logit mortality rates
  Type xq0;
  Type logit_xq0;
  for (int i=0; i<n_obs_direct; i++) {
    int idx_time = time_id_direct(i) - 1;
    log_alpha0 = intercept_log_alpha0 + delta_log_alpha0_c(idx_time) + epsilon_log_alpha0(idx_time);
    log_alpha1 = intercept_log_alpha1 + delta_log_alpha1_c(idx_time) + epsilon_log_alpha1(idx_time);
    log_alpha2 = intercept_log_alpha2 + delta_log_alpha2_c(idx_time) + epsilon_log_alpha2(idx_time);
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
  
  return nll;
}

#undef TMB_OBJECTIVE_PTR
#define TMB_OBJECTIVE_PTR this