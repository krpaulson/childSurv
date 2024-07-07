#undef TMB_OBJECTIVE_PTR
#define TMB_OBJECTIVE_PTR obj

#include "childSurvLL/helpers.hpp"

// survival function
// reference example for syntax: https://kaskr.github.io/adcomp/namespaceromberg.html
template<class Type>
struct survfunc {
  Type log_shape, log_scale;
  survfunc(Type log_shape_, Type log_scale_)
    : log_shape (log_shape_), log_scale (log_scale_) {}
  Type operator()(Type x){
    return 1 / (1 + pow( (x / exp(log_scale)), exp(-1*exp(log_shape))));
  }
};

template <class Type>
Type ll_vr_rw(objective_function<Type>* obj) {

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
  PARAMETER(intercept_log_shape);
  PARAMETER(intercept_log_scale);
  PARAMETER(log_tau_delta_log_shape);
  PARAMETER(log_tau_delta_log_scale);
  PARAMETER(log_tau_epsilon_log_shape);
  PARAMETER(log_tau_epsilon_log_scale);
  //PARAMETER_VECTOR(epsilon);
  //PARAMETER(log_prec_epsilon);
  // PARAMETER(log_phi);
  PARAMETER_VECTOR(delta_log_shape);
  PARAMETER_VECTOR(delta_log_scale);
  PARAMETER_VECTOR(epsilon_log_shape);
  PARAMETER_VECTOR(epsilon_log_scale);
  
  ////////////////////////
  // priors + hyperpriors
  ////////////////////////
  
  // initialize
  Type nll = 0.0;
  
  // intercepts
  nll -= dnorm(intercept_log_shape, Type(0.69), Type(0.35), true);
  nll -= dnorm(intercept_log_scale, Type(38.99), Type(31.78), true);
  
  // RW2 (shape)
  Eigen::SparseMatrix<Type> Q_shape(n_years, n_years);
  for (int i = 0; i < n_years; i++) {
    for (int j = 0; j < n_years; j++) {
      Q_shape.coeffRef(i, j) = R.coeffRef(i, j) * exp(log_tau_delta_log_shape);
    }
  }
  nll += GMRF(Q_shape)(delta_log_shape);
  
  // RW2 (scale)
  Eigen::SparseMatrix<Type> Q_scale(n_years, n_years);
  for (int i = 0; i < n_years; i++) {
    for (int j = 0; j < n_years; j++) {
      Q_scale.coeffRef(i, j) = R.coeffRef(i, j) * exp(log_tau_delta_log_scale);
    }
  }
  nll += GMRF(Q_scale)(delta_log_scale);
  
  // iid effects
  Type sd_epsilon_log_shape = exp(-0.5 * log_tau_epsilon_log_shape);
  Type sd_epsilon_log_scale = exp(-0.5 * log_tau_epsilon_log_scale);
  for (int i = 0; i < n_years; i++) {
    nll -= dnorm(epsilon_log_shape(i), Type(0.0), sd_epsilon_log_shape, true);
    nll -= dnorm(epsilon_log_scale(i), Type(0.0), sd_epsilon_log_scale, true);
  }
  
  // hyperpriors
  //nll -= dpcprec(log_tau_delta_log_shape, Type(0.1), Type(0.5), true); //1.0, 0.01
  //nll -= dpcprec(log_tau_delta_log_scale, Type(0.1), Type(0.5), true); //1.0, 0.01
  nll -= dlgamma(log_tau_delta_log_shape, Type(1.0), Type(1/0.00005), true);
  nll -= dlgamma(log_tau_delta_log_scale, Type(1.0), Type(1/0.00005), true);
  nll -= dlgamma(log_tau_epsilon_log_shape, Type(1.0), Type(1/0.00005), true);
  nll -= dlgamma(log_tau_epsilon_log_scale, Type(1.0), Type(1/0.00005), true);
  
  // overdispersion for Poisson
  /*
  Type sd_epsilon = exp(-0.5 * log_prec_epsilon);
  for (int i = 0; i < n_obs_vr; i++) {
    nll -= dnorm(epsilon(i), Type(0.0), sd_epsilon, true);
  }
  nll -= dnorm(log_prec_epsilon, Type(9.0), Type(3.0), true);
   */
  // nll -= dnorm(log_phi, Type(0.5), Type(1.0), true);
  
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
  matrix<Type> Ax_rw2_shape = (A_rw2_shape * delta_log_shape.matrix());
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
  vector<Type> delta_log_shape_c = delta_log_shape - krig_correct_vec_shape;
  vector<Type> delta_log_scale_c = delta_log_scale - krig_correct_vec_scale;
  
  ////////////////
  // likelihood
  ////////////////
  
  // group 1: nn
  // group 2: pnn if nn and <1 present
  // group 3: <1 if nn not present
  // group 4: 1, 2, 3, 4, or 1-4
  Type log_shape;
  Type log_scale;
  Type lambda;
  Type mx;
  //Type prob;
  //Type phi = exp(log_phi);
  for (int i=0; i<n_obs_vr; i++) {
    int idx_time = time_id_vr(i) - 1;
    log_shape = intercept_log_shape + delta_log_shape_c(idx_time) + epsilon_log_shape(idx_time);
    log_scale = intercept_log_scale + delta_log_scale_c(idx_time) + epsilon_log_scale(idx_time);
    survfunc<Type> s(log_shape, log_scale);
    // nn
    if (months_vr[i] == 0 && n_vr[i] == 1) {
      lambda = 1 - s(Type(1.0));
      nll -= dbinom(obs_vr[i], births[i], lambda, true);
    // pnn
    } else if (months_vr[i] == 1 && n_vr[i] == 11) {
      mx = 12 * (1 - s(Type(12.0))) / romberg::integrate(s, Type(0.0), Type(12.0));
      lambda = (pop[i] * mx - births[i] * (1 - s(Type(1.0)))); // * exp(epsilon[i]);
      if (lambda < 0) {
        lambda = 0;
      }
      nll -= dpois(obs_vr[i], lambda, true);
      //prob = phi / (lambda + phi);
      //nll -= dnbinom(obs_vr[i], phi, prob, true);
    // otherwise
    } else {
      mx = 12 * (s(months_vr[i]) - s(months_vr[i]+n_vr[i])) /
        romberg::integrate(s, months_vr[i], months_vr[i] + n_vr[i]);
      lambda = (pop[i] * mx); // * exp(epsilon[i]);
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
    log_shape = intercept_log_shape + delta_log_shape_c(idx_time) + epsilon_log_shape(idx_time);
    log_scale = intercept_log_scale + delta_log_scale_c(idx_time) + epsilon_log_scale(idx_time);
    logit_xq0 = exp(log_shape) * (log(months_direct(i)) - log_scale);
    nll -= dnorm(obs_direct(i), logit_xq0, se_direct(i), true);
  }
  */
  
  return nll;
}

#undef TMB_OBJECTIVE_PTR
#define TMB_OBJECTIVE_PTR this