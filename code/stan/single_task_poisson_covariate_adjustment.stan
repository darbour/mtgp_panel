data {
  int<lower=1> N;      // number of observations
  int<lower=1> D;      // number of units
  vector[N] x;         // univariate covariate
  matrix[N, D] population;
  array[N * D] int<lower=0> y;         // target variable
  int num_treated;
  array[N * D - num_treated] int control_idx;
  int<lower=1> num_covariates;
  matrix[D, num_covariates] covariates;
}
transformed data {
  // Normalize data
  real xmean = mean(x);
  real xsd = sd(x);
  array[N] real xn = to_array_1d((x - xmean)/xsd);
  real sigma_intercept = 0.1;
  vector[N] jitter = rep_vector(1e-9, N);
}
parameters {
  real<lower=0> lengthscale_global;
  real<lower=0> sigma_global;
  vector[N] z_global;
  row_vector[D] state_offset;
  real intercept;
  array[D] vector[num_covariates] cov_beta;
}
model {
  matrix[N, N] K_global = gp_exp_quad_cov(xn, sigma_global, lengthscale_global);
  matrix[N, N] L_global = cholesky_decompose(add_diag(K_global, jitter));
  row_vector[D] covariate_f;
  for(i in 1:D) {
    cov_beta[i] ~ std_normal();
    covariate_f[i] = covariates[i] * cov_beta[i];
  }
  // priors
  lengthscale_global ~ inv_gamma(5,5); // uniform(0,1);//lognormal(log(.3), .2);
  sigma_global ~ std_normal();
  z_global ~ std_normal();
  state_offset ~ std_normal();

  // index in to only consider the likelihood of the units under control
  y[control_idx] ~ poisson_log(
    intercept + log(to_vector(population)[control_idx]) + to_vector(rep_matrix(covariate_f, N) + rep_matrix(state_offset, N) + rep_matrix(L_global * z_global, D) )[control_idx]);
}
generated quantities {
  matrix[N, D] f;
  int total_obs = N * D - num_treated;
  vector[N * D - num_treated] log_lik;
  matrix[N, D] f_samples;
  row_vector[D] covariate_f;
  for(i in 1:D) {
    covariate_f[i] = covariates[i] * cov_beta[i];
  }
  {
    matrix[N, N] K_global = gp_exp_quad_cov(xn, sigma_global, lengthscale_global);
    matrix[N, N] L_global = cholesky_decompose(add_diag(K_global, jitter));
    // function scaled back to the original scale
    f = (
      intercept + 
      rep_matrix(state_offset, N) + 
      rep_matrix(covariate_f, N) + 
      rep_matrix(L_global * z_global, D) + 
      log(population)
    );
    for(n in 1:total_obs) {
      log_lik[n] = poisson_log_lpmf(y[control_idx[n]] | to_vector(f)[control_idx[n]]);
    }
    for(i in 1:N){
      f_samples[i] = to_row_vector(poisson_log_rng(f[i]));
    }
    f = exp(f);
  }
}
