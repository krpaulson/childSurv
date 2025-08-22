#undef TMB_OBJECTIVE_PTR
#define TMB_OBJECTIVE_PTR obj

#include "childSurv/helpers.hpp"

template <class Type>
Type ll_vr_pspline_poisson_lognormal(objective_function<Type>* obj) {

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
  int n_betas = B.cols();
  DATA_INTEGER(n_obs_direct);
  DATA_IVECTOR(time_id_direct);
  DATA_VECTOR(months_direct);
  DATA_VECTOR(obs_direct);
  DATA_VECTOR(se_direct);
  
  // parameters
  PARAMETER(intercept_log_shape);
  PARAMETER(intercept_log_scale);
  PARAMETER(log_tau_delta_log_shape);
  PARAMETER(log_tau_delta_log_scale);
  PARAMETER(log_tau_epsilon);
  PARAMETER_VECTOR(delta_log_shape);
  PARAMETER_VECTOR(delta_log_scale);
  PARAMETER_VECTOR(epsilon);
  
  
  ////////////////////////
  // priors + hyperpriors
  ////////////////////////
  
  // initialize
  Type nll = 0.0;
  
  // intercepts
  nll -= dnorm(intercept_log_shape, Type(-2.0), Type(0.35), true);
  nll -= dnorm(intercept_log_scale, Type(38.99), Type(31.78), true);
  
  // RW2 (shape)
  Eigen::SparseMatrix<Type> Q_shape(n_betas, n_betas);
  for (int i = 0; i < n_betas; i++) {
    for (int j = 0; j < n_betas; j++) {
      Q_shape.coeffRef(i, j) = R.coeffRef(i, j) * exp(log_tau_delta_log_shape);
    }
  }
  nll += GMRF(Q_shape)(delta_log_shape);
  
  // RW2 (scale)
  Eigen::SparseMatrix<Type> Q_scale(n_betas, n_betas);
  for (int i = 0; i < n_betas; i++) {
    for (int j = 0; j < n_betas; j++) {
      Q_scale.coeffRef(i, j) = R.coeffRef(i, j) * exp(log_tau_delta_log_scale);
    }
  }
  nll += GMRF(Q_scale)(delta_log_scale);
  
  // hyperpriors
  nll -= dpcprec(log_tau_delta_log_shape, Type(0.01), Type(0.5), true);
  nll -= dpcprec(log_tau_delta_log_scale, Type(0.01), Type(0.5), true);
  
  // overdispersion
  Type sd_epsilon = exp(-0.5 * log_tau_epsilon);
  for (int i = 0; i < n_obs_vr; i++) {
    nll -= dnorm(epsilon(i), Type(0.0), sd_epsilon, true);
  }
  nll -= dpcprec(log_tau_epsilon, Type(0.01), Type(0.5), true);
  //nll -= dnorm(log_tau_epsilon, Type(9.2), Type(0.0001), true);
  
  
  ////////////////
  // constraints
  ////////////////
  
  // a constrained vector x_c for an unconstrained vector x is calculated as:
  // x_c = x - Q^{-1}A'(AQ^{-1}A')^{-1}(Ax-e)
  // for constraint Ax=e, and for GMRF x with precision Q
  
  // Invert precision matrices (possible since we added a small value to the diagonal)
  matrix<Type> Q_inv_rw2_shape = invertSparseMatrix(Q_shape);
  matrix<Type> Q_inv_rw2_scale = invertSparseMatrix(Q_scale);
  
  // Create A^T
  matrix<Type> A_rw2_T_shape = A.transpose();
  matrix<Type> A_rw2_T_scale = A.transpose();
  
  // Create Q^{-1}A^T
  matrix<Type> QinvA_rw2_shape = Q_inv_rw2_shape * A_rw2_T_shape;
  matrix<Type> QinvA_rw2_scale = Q_inv_rw2_scale * A_rw2_T_scale;
  
  // Create AQ^{-1}A^T
  matrix<Type> AQinvA_rw2_shape = A * QinvA_rw2_shape;
  matrix<Type> AQinvA_rw2_scale = A * QinvA_rw2_scale;
  
  // Create (AQ^{-1}A^T)^{-1}
  matrix<Type> AQinvA_rw2_inv_shape = AQinvA_rw2_shape.inverse(); // okay for small matrices
  matrix<Type> AQinvA_rw2_inv_scale = AQinvA_rw2_scale.inverse();

  // Create Ax
  matrix<Type> Ax_rw2_shape = (A * delta_log_shape.matrix());
  matrix<Type> Ax_rw2_scale = (A * delta_log_scale.matrix());
  
  // Convert Q^{-1}A'(AQ^{-1}A')^{-1}(Ax-e) to vector form for conditioning by kriging correction
  matrix<Type> krig_correct_shape = QinvA_rw2_shape * AQinvA_rw2_inv_shape * Ax_rw2_shape;
  matrix<Type> krig_correct_scale = QinvA_rw2_scale * AQinvA_rw2_inv_scale * Ax_rw2_scale;
  vector<Type> krig_correct_vec_shape(n_betas);
  vector<Type> krig_correct_vec_scale(n_betas);
  for (int i = 0; i < n_betas; i++) {
    krig_correct_vec_shape(i) = krig_correct_shape(i,0);
    krig_correct_vec_scale(i) = krig_correct_scale(i,0);
  }

  // Construct constrained vector x_c = x - Q^{-1}A'(AQ^{-1}A')^{-1}(Ax-e)
  vector<Type> delta_log_shape_c = delta_log_shape - krig_correct_vec_shape;
  vector<Type> delta_log_scale_c = delta_log_scale - krig_correct_vec_scale;
  
  ////////////////
  // likelihood
  ////////////////
  
  // beta_k = intercept + delta_k
  vector<Type> beta_log_shape(n_betas);
  vector<Type> beta_log_scale(n_betas);
  for (int k = 0; k < n_betas; k++) {
    beta_log_shape(k) =  intercept_log_shape + delta_log_shape_c(k);
    beta_log_scale(k) = intercept_log_scale + delta_log_scale_c(k);
  }

  // group 1: nn
  // group 2: pnn if nn and <1 present
  // group 3: <1 if nn not present
  // group 4: 1, 2, 3, 4, or 1-4
  Type log_shape;
  Type log_scale;
  Type lambda;
  Type mx;
  for (int i=0; i<n_obs_vr; i++) {
    int idx_time = time_id_vr(i) - 1;
    vector<Type> B_t = B.row(idx_time);
    log_shape = (B_t*beta_log_shape).sum();
    log_scale = (B_t*beta_log_scale).sum();
    survfunc<Type> s(log_shape, log_scale);
    // nn
    if (months_vr[i] == 0 && n_vr[i] == 1) {
      lambda = (1 - s(Type(1.0))) * exp(epsilon[i]);
      nll -= dbinom(obs_vr[i], births[i], lambda, true);
    // pnn
    } else if (months_vr[i] == 1 && n_vr[i] == 11) {
      mx = 12 * (1 - s(Type(12.0))) / romberg::integrate(s, Type(0.0), Type(12.0));
      lambda = (pop[i] * mx - births[i] * (1 - s(Type(1.0))));
      lambda = lambda * exp(epsilon[i]);
      if (lambda < 0) {
        lambda = 0;
      }
      nll -= dpois(obs_vr[i], lambda, true);
    // otherwise
    } else {
      mx = 12 * (s(months_vr[i]) - s(months_vr[i]+n_vr[i])) /
        romberg::integrate(s, months_vr[i], months_vr[i] + n_vr[i]);
      lambda = pop[i] * mx * exp(epsilon[i]);
      nll -= dpois(obs_vr[i], lambda, true);
    }
  }
  
  // direct observations of logit mortality rates
  Type logit_xq0;
  for (int i=0; i<n_obs_direct; i++) {
    int idx_time = time_id_direct(i) - 1;
    vector<Type> B_t = B.row(idx_time);
    log_shape = (B_t*beta_log_shape).sum();
    log_scale = (B_t*beta_log_scale).sum();
    logit_xq0 = exp(-1*exp(log_shape)) * (log(months_direct(i)) - log_scale);
    nll -= dnorm(obs_direct(i), logit_xq0, se_direct(i), true);
  }
  
  return nll;
}

#undef TMB_OBJECTIVE_PTR
#define TMB_OBJECTIVE_PTR this