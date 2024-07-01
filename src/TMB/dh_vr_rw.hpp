#undef TMB_OBJECTIVE_PTR
#define TMB_OBJECTIVE_PTR obj

#include "childSurvLL/helpers.hpp"

template <class Type>
Type dh_vr_rw(objective_function<Type>* obj) {

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
  /*
  DATA_INTEGER(n_obs_direct);
  DATA_IVECTOR(time_id_direct);
  DATA_VECTOR(months_direct);
  DATA_VECTOR(obs_direct);
  DATA_VECTOR(se_direct);
   */
  
  // parameters
  PARAMETER(intercept_logit_B0);
  PARAMETER(intercept_logit_B1);
  PARAMETER(intercept_logit_B2);
  PARAMETER(log_tau_delta_logit_B0);
  PARAMETER(log_tau_delta_logit_B1);
  PARAMETER(log_tau_delta_logit_B2);
  PARAMETER(log_tau_epsilon_logit_B0);
  PARAMETER(log_tau_epsilon_logit_B1);
  PARAMETER(log_tau_epsilon_logit_B2);
  //PARAMETER(log_phi);
  PARAMETER_VECTOR(delta_logit_B0);
  PARAMETER_VECTOR(delta_logit_B1);
  PARAMETER_VECTOR(delta_logit_B2);
  PARAMETER_VECTOR(epsilon_logit_B0);
  PARAMETER_VECTOR(epsilon_logit_B1);
  PARAMETER_VECTOR(epsilon_logit_B2);
  
  ////////////////////////
  // priors + hyperpriors
  ////////////////////////
  
  // initialize
  Type nll = 0.0;
  
  // intercepts
  nll -= dnorm(intercept_logit_B0, Type(0.0), Type(30.00), true);
  nll -= dnorm(intercept_logit_B1, Type(0.0), Type(30.00), true);
  nll -= dnorm(intercept_logit_B2, Type(0.0), Type(30.00), true);
  
  // RW2 (Beta0)
  Eigen::SparseMatrix<Type> Q0(n_years, n_years);
  for (int i = 0; i < n_years; i++) {
    for (int j = 0; j < n_years; j++) {
      Q0.coeffRef(i, j) = R.coeffRef(i, j) * exp(log_tau_delta_logit_B0);
    }
  }
  nll += GMRF(Q0)(delta_logit_B0);
  
  // RW2 (Beta1)
  Eigen::SparseMatrix<Type> Q1(n_years, n_years);
  for (int i = 0; i < n_years; i++) {
    for (int j = 0; j < n_years; j++) {
      Q1.coeffRef(i, j) = R.coeffRef(i, j) * exp(log_tau_delta_logit_B1);
    }
  }
  nll += GMRF(Q1)(delta_logit_B1);
  
  // RW2 (Beta2)
  Eigen::SparseMatrix<Type> Q2(n_years, n_years);
  for (int i = 0; i < n_years; i++) {
    for (int j = 0; j < n_years; j++) {
      Q2.coeffRef(i, j) = R.coeffRef(i, j) * exp(log_tau_delta_logit_B2);
    }
  }
  nll += GMRF(Q2)(delta_logit_B2);
  
  // iid effects
  Type sd_epsilon_logit_B0 = exp(-0.5 * log_tau_epsilon_logit_B0);
  Type sd_epsilon_logit_B1 = exp(-0.5 * log_tau_epsilon_logit_B1);
  Type sd_epsilon_logit_B2 = exp(-0.5 * log_tau_epsilon_logit_B2);
  for (int i = 0; i < n_years; i++) {
    nll -= dnorm(epsilon_logit_B0(i), Type(0.0), sd_epsilon_logit_B0, true);
    nll -= dnorm(epsilon_logit_B1(i), Type(0.0), sd_epsilon_logit_B1, true);
    nll -= dnorm(epsilon_logit_B2(i), Type(0.0), sd_epsilon_logit_B2, true);
  }
  
  // hyperpriors
  //nll -= dpcprec(log_tau_delta_log_B0, Type(0.001), Type(0.5), true);
  //nll -= dpcprec(log_tau_delta_log_B1, Type(0.001), Type(0.5), true);
  //nll -= dpcprec(log_tau_delta_log_B2, Type(0.001), Type(0.5), true);
  nll -= dlgamma(log_tau_delta_logit_B0, Type(1.0), Type(1/0.00005), true);
  nll -= dlgamma(log_tau_delta_logit_B1, Type(1.0), Type(1/0.00005), true);
  nll -= dlgamma(log_tau_delta_logit_B2, Type(1.0), Type(1/0.00005), true);
  nll -= dlgamma(log_tau_epsilon_logit_B0, Type(1.0), Type(1/0.00005), true);
  nll -= dlgamma(log_tau_epsilon_logit_B1, Type(1.0), Type(1/0.00005), true);
  nll -= dlgamma(log_tau_epsilon_logit_B2, Type(1.0), Type(1/0.00005), true);
  
  // overdispersion parameter (negative binomial)
  //nll -= dnorm(log_phi, Type(0.0), Type(30.0), true);
  
  ////////////////
  // constraints
  ////////////////
  
  // a constrained vector x_c for an unconstrained vector x is calculated as:
  // x_c = x - Q^{-1}A'(AQ^{-1}A')^{-1}(Ax-e)
  // for constraint Ax=e, and for GMRF x with precision Q
  
  // Invert precision matrices (possible since we added a small value to the diagonal)
  matrix<Type> Q_inv_rw2_B0 = invertSparseMatrix(Q0);
  matrix<Type> Q_inv_rw2_B1 = invertSparseMatrix(Q1);
  matrix<Type> Q_inv_rw2_B2 = invertSparseMatrix(Q2);
  
  // Create constraint matrices A
  matrix<Type> A_rw2_B0(1, n_years);
  matrix<Type> A_rw2_B1(1, n_years);
  matrix<Type> A_rw2_B2(1, n_years);
  for(int i = 0; i < n_years; i++) {
    A_rw2_B0(0, i) = 1; // sum-to-0 constraint
    A_rw2_B1(0, i) = 1;
    A_rw2_B2(0, i) = 1;
  }
  
  // Create A^T
  matrix<Type> A_rw2_T_B0 = A_rw2_B0.transpose();
  matrix<Type> A_rw2_T_B1 = A_rw2_B1.transpose();
  matrix<Type> A_rw2_T_B2 = A_rw2_B2.transpose();
  
  // Create Q^{-1}A^T
  matrix<Type> QinvA_rw2_B0 = Q_inv_rw2_B0 * A_rw2_T_B0;
  matrix<Type> QinvA_rw2_B1 = Q_inv_rw2_B1 * A_rw2_T_B1;
  matrix<Type> QinvA_rw2_B2 = Q_inv_rw2_B2 * A_rw2_T_B2;
  
  // Create AQ^{-1}A^T
  matrix<Type> AQinvA_rw2_B0 = A_rw2_B0 * QinvA_rw2_B0;
  matrix<Type> AQinvA_rw2_B1 = A_rw2_B1 * QinvA_rw2_B1;
  matrix<Type> AQinvA_rw2_B2 = A_rw2_B2 * QinvA_rw2_B2;
  
  // Create (AQ^{-1}A^T)^{-1}
  matrix<Type> AQinvA_rw2_inv_B0 = AQinvA_rw2_B0.inverse(); // okay for small matrices
  matrix<Type> AQinvA_rw2_inv_B1 = AQinvA_rw2_B1.inverse();
  matrix<Type> AQinvA_rw2_inv_B2 = AQinvA_rw2_B2.inverse();
  
  // Create Ax
  matrix<Type> Ax_rw2_B0 = (A_rw2_B0 * delta_logit_B0.matrix());
  matrix<Type> Ax_rw2_B1 = (A_rw2_B1 * delta_logit_B1.matrix());
  matrix<Type> Ax_rw2_B2 = (A_rw2_B2 * delta_logit_B2.matrix());
  
  // Convert Ax from matrix to vector form - needed for dnorm & MVNORM
  vector<Type> Ax_rw2_vec_B0(1);
  vector<Type> Ax_rw2_vec_B1(1);
  vector<Type> Ax_rw2_vec_B2(1);
  for(int i = 0; i < 1; i++) {
    Ax_rw2_vec_B0(i) = Ax_rw2_B0(i,0);
    Ax_rw2_vec_B1(i) = Ax_rw2_B1(i,0);
    Ax_rw2_vec_B2(i) = Ax_rw2_B2(i,0);
  }
  
  // Convert Q^{-1}A'(AQ^{-1}A')^{-1}(Ax-e) to vector form for conditioning by kriging correction
  matrix<Type> krig_correct_B0 = QinvA_rw2_B0 * AQinvA_rw2_inv_B0 * Ax_rw2_B0;
  matrix<Type> krig_correct_B1 = QinvA_rw2_B1 * AQinvA_rw2_inv_B1 * Ax_rw2_B1;
  matrix<Type> krig_correct_B2 = QinvA_rw2_B2 * AQinvA_rw2_inv_B2 * Ax_rw2_B2;
  vector<Type> krig_correct_vec_B0(n_years);
  vector<Type> krig_correct_vec_B1(n_years);
  vector<Type> krig_correct_vec_B2(n_years);
  for (int i = 0; i < n_years; i++) {
    krig_correct_vec_B0(i) = krig_correct_B0(i,0);
    krig_correct_vec_B1(i) = krig_correct_B1(i,0);
    krig_correct_vec_B2(i) = krig_correct_B2(i,0);
  }
  
  // Construct constrained vector x_c = x - Q^{-1}A'(AQ^{-1}A')^{-1}(Ax-e)
  vector<Type> delta_logit_B0_c = delta_logit_B0 - krig_correct_vec_B0;
  vector<Type> delta_logit_B1_c = delta_logit_B1 - krig_correct_vec_B1;
  vector<Type> delta_logit_B2_c = delta_logit_B2 - krig_correct_vec_B2;
  
  ////////////////
  // likelihood
  ////////////////
  
  Type logit_B0;
  Type logit_B1;
  Type logit_B2;
  Type B0;
  Type B1;
  Type B2;
  Type lambda;
  Type mx;
  Type prob;
  //Type phi = exp(log_phi);
  for (int i=0; i<n_obs_vr; i++) {
    int idx_time = time_id_vr(i) - 1;
    logit_B0 = intercept_logit_B0 + delta_logit_B0_c(idx_time) + epsilon_logit_B0(idx_time);
    logit_B1 = intercept_logit_B1 + delta_logit_B1_c(idx_time) + epsilon_logit_B1(idx_time);
    logit_B2 = intercept_logit_B2 + delta_logit_B2_c(idx_time) + epsilon_logit_B2(idx_time);
    B0 = exp(logit_B0) / (1 + exp(logit_B0));
    B1 = exp(logit_B1) / (1 + exp(logit_B1));
    B2 = exp(logit_B2) / (1 + exp(logit_B2));
    // nn
    if (months_vr[i] == 0 && n_vr[i] == 1) {
      lambda = 1 - exp(-1 * B0);
      nll -= dbinom(obs_vr[i], births[i], lambda, true);
    // pnn
    } else if (months_vr[i] == 1 && n_vr[i] == 11) {
      mx = 12 * (1 - exp(-1 * (B0 + 11*B1))) /
        ( (1-exp(-1*B0))/B0 + exp(-1*(B0 - B1)) * ( (exp(-1*B1) - exp(-12*B1))/B1 ) );
      lambda = (pop[i] * mx - births[i] * (1 - exp(-1 * B0)));
      if (lambda < 0) {
        lambda = 0;
      }
      nll -= dpois(obs_vr[i], lambda, true);
      //prob = phi / (lambda + phi);
      //nll -= dnbinom(obs_vr[i], phi, prob, true);
    // infant
    } else if (months_vr[i] == 0 && n_vr[i] == 12) {
      mx = 12 * (1 - exp(-1 * (B0 + 11*B1))) /
        ( (1-exp(-1*B0))/B0 + exp(-1*(B0 - B1)) * ( (exp(-1*B1) - exp(-12*B1))/B1 ) );
      lambda = (pop[i] * mx);
      nll -= dpois(obs_vr[i], lambda, true);
      //prob = phi / (lambda + phi);
      //nll -= dnbinom(obs_vr[i], phi, prob, true);
    // 1-4 years
    } else {
      mx = 12 * (1-exp(-48*B2)) /
        ( exp(12*B2) * ((exp(-12*B2) - exp(-60*B2)))/B2 );
      lambda = (pop[i] * mx);
      nll -= dpois(obs_vr[i], lambda, true);
      //prob = phi / (lambda + phi);
      //nll -= dnbinom(obs_vr[i], phi, prob, true);
    }
  }
  
  // direct observations of logit mortality rates
  /*
  Type logit_xq0;
  for (int i=0; i<n_obs_direct; i++) {
    int idx_time = time_id_direct(i) - 1;
    log_B0 = intercept_log_B0 + delta_log_B0_c(idx_time) + epsilon_log_B0(idx_time);
    log_B1 = intercept_log_B1 + delta_log_B1_c(idx_time) + epsilon_log_B1(idx_time);
    log_B2 = intercept_log_B2 + delta_log_B2_c(idx_time) + epsilon_log_B2(idx_time);
    logit_xq0 = exp(log_shape) * (log(months_direct(i)) - log_scale); // need to update this for DH
    nll -= dnorm(obs_direct(i), logit_xq0, se_direct(i), true);
  }
  */
  
  return nll;
}

#undef TMB_OBJECTIVE_PTR
#define TMB_OBJECTIVE_PTR this