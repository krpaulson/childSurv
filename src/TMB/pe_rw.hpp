#undef TMB_OBJECTIVE_PTR
#define TMB_OBJECTIVE_PTR obj

#include "childSurv/helpers.hpp"

template <class Type>
Type pe_rw(objective_function<Type>* obj) {

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
  PARAMETER(intercept_log_alpha1);
  PARAMETER(intercept_log_alpha2);
  PARAMETER(intercept_log_alpha3);
  PARAMETER(log_tau_delta_log_alpha1);
  PARAMETER(log_tau_delta_log_alpha2);
  PARAMETER(log_tau_delta_log_alpha3);
  PARAMETER(log_tau_epsilon_log_alpha1);
  PARAMETER(log_tau_epsilon_log_alpha2);
  PARAMETER(log_tau_epsilon_log_alpha3);
  PARAMETER_VECTOR(delta_log_alpha1);
  PARAMETER_VECTOR(delta_log_alpha2);
  PARAMETER_VECTOR(delta_log_alpha3);
  PARAMETER_VECTOR(epsilon_log_alpha1);
  PARAMETER_VECTOR(epsilon_log_alpha2);
  PARAMETER_VECTOR(epsilon_log_alpha3);
  
  ////////////////////////
  // priors + hyperpriors
  ////////////////////////
  
  // initialize
  Type nll = 0.0;
  
  // intercepts
  nll -= dnorm(intercept_log_alpha1, Type(0.0), Type(30.00), true);
  nll -= dnorm(intercept_log_alpha2, Type(0.0), Type(30.00), true);
  nll -= dnorm(intercept_log_alpha3, Type(0.0), Type(30.00), true);
  
  // RW2 (alpha1)
  Eigen::SparseMatrix<Type> Q1(n_years, n_years);
  for (int i = 0; i < n_years; i++) {
    for (int j = 0; j < n_years; j++) {
      Q1.coeffRef(i, j) = R.coeffRef(i, j) * exp(log_tau_delta_log_alpha1);
    }
  }
  nll += GMRF(Q1)(delta_log_alpha1);
  
  // RW2 (alpha2)
  Eigen::SparseMatrix<Type> Q2(n_years, n_years);
  for (int i = 0; i < n_years; i++) {
    for (int j = 0; j < n_years; j++) {
      Q2.coeffRef(i, j) = R.coeffRef(i, j) * exp(log_tau_delta_log_alpha2);
    }
  }
  nll += GMRF(Q2)(delta_log_alpha2);
  
  // RW2 (alpha3)
  Eigen::SparseMatrix<Type> Q3(n_years, n_years);
  for (int i = 0; i < n_years; i++) {
    for (int j = 0; j < n_years; j++) {
      Q3.coeffRef(i, j) = R.coeffRef(i, j) * exp(log_tau_delta_log_alpha3);
    }
  }
  nll += GMRF(Q3)(delta_log_alpha3);
  
  // iid effects
  Type sd_epsilon_log_alpha1 = exp(-0.5 * log_tau_epsilon_log_alpha1);
  Type sd_epsilon_log_alpha2 = exp(-0.5 * log_tau_epsilon_log_alpha2);
  Type sd_epsilon_log_alpha3 = exp(-0.5 * log_tau_epsilon_log_alpha3);
  for (int i = 0; i < n_years; i++) {
    nll -= dnorm(epsilon_log_alpha1(i), Type(0.0), sd_epsilon_log_alpha1, true);
    nll -= dnorm(epsilon_log_alpha2(i), Type(0.0), sd_epsilon_log_alpha2, true);
    nll -= dnorm(epsilon_log_alpha3(i), Type(0.0), sd_epsilon_log_alpha3, true);
  }
  
  // hyperpriors
  nll -= dpcprec(log_tau_delta_log_alpha1, Type(0.001), Type(0.5), true);
  nll -= dpcprec(log_tau_delta_log_alpha2, Type(0.001), Type(0.5), true);
  nll -= dpcprec(log_tau_delta_log_alpha3, Type(0.001), Type(0.5), true);
  //nll -= dlgamma(log_tau_delta_log_alpha0, Type(1.0), Type(1/0.00005), true);
  //nll -= dlgamma(log_tau_delta_log_alpha1, Type(1.0), Type(1/0.00005), true);
  //nll -= dlgamma(log_tau_delta_log_alpha2, Type(1.0), Type(1/0.00005), true);
  nll -= dlgamma(log_tau_epsilon_log_alpha1, Type(1.0), Type(1/0.00005), true);
  nll -= dlgamma(log_tau_epsilon_log_alpha2, Type(1.0), Type(1/0.00005), true);
  nll -= dlgamma(log_tau_epsilon_log_alpha3, Type(1.0), Type(1/0.00005), true);
  
  ////////////////
  // constraints
  ////////////////
  
  // a constrained vector x_c for an unconstrained vector x is calculated as:
  // x_c = x - Q^{-1}A'(AQ^{-1}A')^{-1}(Ax-e)
  // for constraint Ax=e, and for GMRF x with precision Q
  
  // Invert precision matrices (possible since we added a small value to the diagonal)
  matrix<Type> Q_inv_rw2_log_alpha1 = invertSparseMatrix(Q1);
  matrix<Type> Q_inv_rw2_log_alpha2 = invertSparseMatrix(Q2);
  matrix<Type> Q_inv_rw2_log_alpha3 = invertSparseMatrix(Q3);
  
  // Create constraint matrices A
  matrix<Type> A_rw2_log_alpha1(1, n_years);
  matrix<Type> A_rw2_log_alpha2(1, n_years);
  matrix<Type> A_rw2_log_alpha3(1, n_years);
  for(int i = 0; i < n_years; i++) {
    A_rw2_log_alpha1(0, i) = 1; // sum-to-0 constraint
    A_rw2_log_alpha2(0, i) = 1;
    A_rw2_log_alpha3(0, i) = 1;
  }
  
  // Create A^T
  matrix<Type> A_rw2_T_log_alpha1 = A_rw2_log_alpha1.transpose();
  matrix<Type> A_rw2_T_log_alpha2 = A_rw2_log_alpha2.transpose();
  matrix<Type> A_rw2_T_log_alpha3 = A_rw2_log_alpha3.transpose();
  
  // Create Q^{-1}A^T
  matrix<Type> QinvA_rw2_log_alpha1 = Q_inv_rw2_log_alpha1 * A_rw2_T_log_alpha1;
  matrix<Type> QinvA_rw2_log_alpha2 = Q_inv_rw2_log_alpha2 * A_rw2_T_log_alpha2;
  matrix<Type> QinvA_rw2_log_alpha3 = Q_inv_rw2_log_alpha3 * A_rw2_T_log_alpha3;
  
  // Create AQ^{-1}A^T
  matrix<Type> AQinvA_rw2_log_alpha1 = A_rw2_log_alpha1 * QinvA_rw2_log_alpha1;
  matrix<Type> AQinvA_rw2_log_alpha2 = A_rw2_log_alpha2 * QinvA_rw2_log_alpha2;
  matrix<Type> AQinvA_rw2_log_alpha3 = A_rw2_log_alpha3 * QinvA_rw2_log_alpha3;
  
  // Create (AQ^{-1}A^T)^{-1}
  matrix<Type> AQinvA_rw2_inv_log_alpha1 = AQinvA_rw2_log_alpha1.inverse(); // okay for small matrices
  matrix<Type> AQinvA_rw2_inv_log_alpha2 = AQinvA_rw2_log_alpha2.inverse();
  matrix<Type> AQinvA_rw2_inv_log_alpha3 = AQinvA_rw2_log_alpha3.inverse();
  
  // Create Ax
  matrix<Type> Ax_rw2_log_alpha1 = (A_rw2_log_alpha1 * delta_log_alpha1.matrix());
  matrix<Type> Ax_rw2_log_alpha2 = (A_rw2_log_alpha2 * delta_log_alpha2.matrix());
  matrix<Type> Ax_rw2_log_alpha3 = (A_rw2_log_alpha3 * delta_log_alpha3.matrix());
  
  // Convert Ax from matrix to vector form - needed for dnorm & MVNORM
  vector<Type> Ax_rw2_vec_log_alpha1(1);
  vector<Type> Ax_rw2_vec_log_alpha2(1);
  vector<Type> Ax_rw2_vec_log_alpha3(1);
  for(int i = 0; i < 1; i++) {
    Ax_rw2_vec_log_alpha1(i) = Ax_rw2_log_alpha1(i,0);
    Ax_rw2_vec_log_alpha2(i) = Ax_rw2_log_alpha2(i,0);
    Ax_rw2_vec_log_alpha3(i) = Ax_rw2_log_alpha3(i,0);
  }
  
  // Convert Q^{-1}A'(AQ^{-1}A')^{-1}(Ax-e) to vector form for conditioning by kriging correction
  matrix<Type> krig_correct_log_alpha1 = QinvA_rw2_log_alpha1 * AQinvA_rw2_inv_log_alpha1 * Ax_rw2_log_alpha1;
  matrix<Type> krig_correct_log_alpha2 = QinvA_rw2_log_alpha2 * AQinvA_rw2_inv_log_alpha2 * Ax_rw2_log_alpha2;
  matrix<Type> krig_correct_log_alpha3 = QinvA_rw2_log_alpha3 * AQinvA_rw2_inv_log_alpha3 * Ax_rw2_log_alpha3;
  vector<Type> krig_correct_vec_log_alpha1(n_years);
  vector<Type> krig_correct_vec_log_alpha2(n_years);
  vector<Type> krig_correct_vec_log_alpha3(n_years);
  for (int i = 0; i < n_years; i++) {
    krig_correct_vec_log_alpha1(i) = krig_correct_log_alpha1(i,0);
    krig_correct_vec_log_alpha2(i) = krig_correct_log_alpha2(i,0);
    krig_correct_vec_log_alpha3(i) = krig_correct_log_alpha3(i,0);
  }
  
  // Construct constrained vector x_c = x - Q^{-1}A'(AQ^{-1}A')^{-1}(Ax-e)
  vector<Type> delta_log_alpha1_c = delta_log_alpha1 - krig_correct_vec_log_alpha1;
  vector<Type> delta_log_alpha2_c = delta_log_alpha2 - krig_correct_vec_log_alpha2;
  vector<Type> delta_log_alpha3_c = delta_log_alpha3 - krig_correct_vec_log_alpha3;
  
  ////////////////
  // likelihood
  ////////////////
  
  // pseudo-likelihood estimates from FBH
  vector<Type> x(n_obs_fbh);
  Type theta;
  for(int i = 0; i < n_obs_fbh; i++){
    int idx_time = time_id_fbh(i) - 1; // fetch time index for this observation (0-indexing)
    if(par_id_fbh(i) == 1) {
      theta = intercept_log_alpha1 + delta_log_alpha1_c(idx_time) + epsilon_log_alpha1(idx_time);
    }
    if(par_id_fbh(i) == 2) {
      theta = intercept_log_alpha2 + delta_log_alpha2_c(idx_time) + epsilon_log_alpha2(idx_time);
    }
    if(par_id_fbh(i) == 3) {
      theta = intercept_log_alpha3 + delta_log_alpha3_c(idx_time) + epsilon_log_alpha3(idx_time);
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
  Type log_alpha1;
  Type log_alpha2;
  Type log_alpha3;
  Type alpha1;
  Type alpha2;
  Type alpha3;
  Type lambda;
  Type mx;
  Type prob;
  for (int i=0; i<n_obs_vr; i++) {
    int idx_time = time_id_vr(i) - 1;
    log_alpha1 = intercept_log_alpha1 + delta_log_alpha1_c(idx_time) + epsilon_log_alpha1(idx_time);
    log_alpha2 = intercept_log_alpha2 + delta_log_alpha2_c(idx_time) + epsilon_log_alpha2(idx_time);
    log_alpha3 = intercept_log_alpha3 + delta_log_alpha3_c(idx_time) + epsilon_log_alpha3(idx_time);
    alpha1 = exp(log_alpha1);
    alpha2 = exp(log_alpha2);
    alpha3 = exp(log_alpha3);
    // nn
    if (months_vr[i] == 0 && n_vr[i] == 1) {
      lambda = 1 - exp(-1 * (alpha1 + alpha2 + alpha3));
      nll -= dbinom(obs_vr[i], births[i], lambda, true);
    // pnn
    } else if (months_vr[i] == 1 && n_vr[i] == 11) {
      mx = 12 * (1 - exp(-1 * (12*alpha1 + 12*alpha2 + alpha3))) /
        ((1 - exp(-1 * (alpha1 + alpha2 + alpha3))) / (alpha1 + alpha2 + alpha3) +
          ((exp(-1 * (alpha1 + alpha2 + alpha3)) - exp(-1 * (12*alpha1+ 12*alpha2 + alpha3))) / (alpha1 + alpha2)));
      lambda = (pop[i] * mx - births[i] * (1 - exp(-1 * (alpha1 + alpha2 + alpha3))));
      if (lambda < 0) {
        lambda = 0;
      }
      nll -= dpois(obs_vr[i], lambda, true);
    // infant
    } else if (months_vr[i] == 0 && n_vr[i] == 12) {
      mx = 12 * (1 - exp(-1 * (12*alpha1 + 12*alpha2 + alpha3))) /
        ((1 - exp(-1 * (alpha1 + alpha2 + alpha3))) / (alpha1 + alpha2 + alpha3) +
          ((exp(-1 * (alpha1 + alpha2 + alpha3)) - exp(-1 * (12*alpha1 + 12*alpha2 + alpha3))) / (alpha1 + alpha2)));
      lambda = (pop[i] * mx);
      nll -= dpois(obs_vr[i], lambda, true);
    // 1-4 years
    } else {
      mx = 12 * alpha1;
      lambda = (pop[i] * mx);
      nll -= dpois(obs_vr[i], lambda, true);
    }
    // Rcout << "Likelihood for obs " << i << ": " << nll; // for debugging
  }
  
  // pre-processed observations of logit mortality rates
  Type xq0;
  Type logit_xq0;
  for (int i=0; i<n_obs_pp; i++) {
    int idx_time = time_id_pp(i) - 1;
    log_alpha1 = intercept_log_alpha1 + delta_log_alpha1_c(idx_time) + epsilon_log_alpha1(idx_time);
    log_alpha2 = intercept_log_alpha2 + delta_log_alpha2_c(idx_time) + epsilon_log_alpha2(idx_time);
    log_alpha3 = intercept_log_alpha3 + delta_log_alpha3_c(idx_time) + epsilon_log_alpha3(idx_time);
    if (months_pp[i] == 1) {
      xq0 = 1 - exp(-1 * (exp(log_alpha1) + exp(log_alpha2) + exp(log_alpha3)));
    } else if (months_pp[i] == 12) {
      xq0 = 1 - exp(-1 * (12*exp(log_alpha1) + 12*exp(log_alpha2) + exp(log_alpha3)));
    } else if (months_pp[i] == 60) {
      xq0 = 1 - exp(-1 * (60*exp(log_alpha1) + 12*exp(log_alpha2) + exp(log_alpha3)));
    }
    logit_xq0 = log(xq0 / (1-xq0));;
    nll -= dnorm(obs_pp(i), logit_xq0, se_pp(i), true);
  }
  
  return nll;
}

#undef TMB_OBJECTIVE_PTR
#define TMB_OBJECTIVE_PTR this