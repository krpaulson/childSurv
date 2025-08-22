
#ifndef helpers_hpp
#define helpers_hpp 1

// pc prior for normal precisions
// https://github.com/taylorokonek/stbench/blob/main/inst/include/stbench/addtl_densities.hpp
template<class Type>
Type dpcprec(Type log_tau, Type U, Type alpha, int give_log = 0) {
  Type lambda = -log(alpha) / U;
  Type logres = log(lambda) - log(2) - (3/2)*log_tau - lambda * pow(exp(log_tau), -1/2);
  if(give_log) return logres; else return exp(logres);
};

// survival function for log-logistic model
// reference example for syntax: https://kaskr.github.io/adcomp/namespaceromberg.html
template<class Type>
struct survfunc {
  Type logit_shape, log_scale;
  survfunc(Type logit_shape_, Type log_scale_)
    : logit_shape (logit_shape_), log_scale (log_scale_) {}
  Type operator()(Type x){
    return 1 / (1 + pow( (x / exp(log_scale)), exp(logit_shape)/(1+exp(logit_shape)) ));
  }
};

#endif

