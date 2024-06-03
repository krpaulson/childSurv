

// pc prior for normal precisions
// https://github.com/taylorokonek/stbench/blob/main/inst/include/stbench/addtl_densities.hpp
template<class Type>
Type dpcprec(Type log_tau, Type U, Type alpha, int give_log = 0) {
  Type lambda = -log(alpha) / U;
  Type logres = log(lambda) - log(2) - log_tau/2 - lambda * pow(exp(log_tau), -1/2);
  if(give_log) return logres; else return exp(logres);
}