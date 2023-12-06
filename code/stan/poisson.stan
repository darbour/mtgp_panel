data {
  int<lower=1> N;      // number of observations
  int<lower=1> D;      // number of units
  int<lower=1> n_k_f;      // number of latent functions for f
  vector[N] x;         // univariate covariate
  matrix[N, D] population;
  array[N * D] int<lower=0> y;         // target variable
  int num_treated;
  array[N * D - num_treated] int control_idx;
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

  real<lower=0> lengthscale_f; // lengthscale of f
  real<lower=0> sigma_f;       // scale of f
  matrix[N, n_k_f] z_f;
  matrix[n_k_f, D] k_f;
  real intercept;
}
model {
  // covariances and Cholesky decompositions
  matrix[N, N] K_f = gp_exp_quad_cov(xn, sigma_f, lengthscale_f);
  matrix[N, N] L_f = cholesky_decompose(add_diag(K_f, jitter));

  matrix[N, N] K_global = gp_exp_quad_cov(xn, sigma_global, lengthscale_global);
  matrix[N, N] L_global = cholesky_decompose(add_diag(K_global, jitter));

  // priors
  to_vector(z_f) ~ std_normal();
  to_vector(k_f) ~ std_normal();
  lengthscale_f ~ inv_gamma(5, 5); // uniform(0,1);//lognormal(log(.3), .2);
  sigma_f ~ std_normal();

  lengthscale_global ~ inv_gamma(5,5); // uniform(0,1);//lognormal(log(.3), .2);
  sigma_global ~ std_normal();
  z_global ~ std_normal();
  state_offset ~ std_normal();

  // index in to only consider the likelihood of the units under control
  y[control_idx] ~ poisson_log(
    intercept + log(to_vector(population)[control_idx]) + to_vector(rep_matrix(state_offset, N) + rep_matrix(L_global * z_global, D) + L_f * z_f * k_f)[control_idx]);
}
generated quantities {
  matrix[N, D] f;
  matrix[N, D] f_samples;
  int total_obs = N * D - num_treated;
  vector[N * D - num_treated] log_lik;
  {
    // covariances and Cholesky decompositions
    matrix[N, N] K_f = gp_exp_quad_cov(xn, sigma_f, lengthscale_f);
    matrix[N, N] L_f = cholesky_decompose(add_diag(K_f, jitter));

    matrix[N, N] K_global = gp_exp_quad_cov(xn, sigma_global, lengthscale_global);
    matrix[N, N] L_global = cholesky_decompose(add_diag(K_global, jitter));
    // function scaled back to the original scale
    f = (intercept + rep_matrix(state_offset, N) + rep_matrix(L_global * z_global, D) + log(population) + L_f * z_f * k_f);
    for(n in 1:total_obs) {
      log_lik[n] = poisson_log_lpmf(y[control_idx[n]] | to_vector(f)[control_idx[n]]);

    }
    for(i in 1:N){
      f_samples[i] = to_row_vector(poisson_log_rng(f[i]));
    }
    f = exp(f);
  }
}
