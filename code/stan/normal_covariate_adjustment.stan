data {
  int<lower=1> N;      // number of observations
  int<lower=1> D;      // number of units
  int<lower=1> n_k_f;      // number of latent functions for f
  vector[N] x;         // univariate covariate
  matrix[N, D] y;         // target variable
  matrix[N, D] inv_population;
  int num_treated;
  int control_idx[N * D - num_treated];
  int<lower=1> num_covariates;
  matrix[D, num_covariates] covariates;
}
transformed data {
  // Normalize data
  real xmean = mean(x);
  real xsd = sd(x);
  real xn[N] = to_array_1d((x - xmean)/xsd);
  real sigma_intercept = 0.1;
  vector[N] jitter = rep_vector(1e-9, N);
}
parameters {
  real<lower=0> lengthscale_global;
  real<lower=0> sigma_global;
  real<lower=0> lengthscale_f; // lengthscale of f
  real<lower=0> sigma_f;       // scale of f
  real<lower=0> sigman;
  vector[num_covariates] cov_beta[D];
  vector[D] state_offset;
  vector[N] z_global;
  matrix[N, n_k_f] z_f;
  matrix[n_k_f, D] k_f;
  real global_offset;
}

model {
  // covariances and Cholesky decompositions
  matrix[N, N] K_f = gp_exp_quad_cov(xn, sigma_f, lengthscale_f);
  matrix[N, N] L_f = cholesky_decompose(add_diag(K_f, jitter));

  matrix[N, N] K_global = gp_exp_quad_cov(xn, sigma_global, lengthscale_global);
  matrix[N, N] L_global = cholesky_decompose(add_diag(K_global, jitter));
  vector[D] covariate_f;
  for(i in 1:D) {
    cov_beta[i] ~ std_normal();
    covariate_f[i] = covariates[i] * cov_beta[i];
  }

  // priors
  to_vector(z_f) ~ std_normal();
  to_vector(k_f) ~ std_normal();
  z_global ~ std_normal();
  lengthscale_f ~ inv_gamma(5,5);
  lengthscale_global ~ inv_gamma(5,5);
  sigma_f ~ normal(0, 1);
  sigma_global ~ normal(0, 1);
  sigman ~ normal(0, 1);
  state_offset ~ normal(0, 1);

  // index in to only consider the likelihood of the units under control
  to_vector(y)[control_idx] ~ normal(
      to_vector(
        global_offset +
        rep_matrix(state_offset, N)' +
        rep_matrix(covariate_f, N)' +
        rep_matrix(L_global * z_global, D) +
        L_f * z_f * k_f
      )[control_idx],
      sigman * sqrt(to_vector(inv_population))[control_idx]
  );
}
generated quantities {
  matrix[N, D] f;
  matrix[N, D] f_samples;
  int total_obs = N * D - num_treated;
  vector[N * D - num_treated] log_lik;

  matrix[N, N] K_f = gp_exp_quad_cov(xn, sigma_f, lengthscale_f);
  matrix[N, N] L_f = cholesky_decompose(add_diag(K_f, jitter));

  matrix[N, N] K_global = gp_exp_quad_cov(xn, sigma_global, lengthscale_global);
  matrix[N, N] L_global = cholesky_decompose(add_diag(K_global, jitter));
  vector[D] covariate_f;
  
  for(i in 1:D) {
    covariate_f[i] = covariates[i] * cov_beta[i];
  }
  // check priors for sigma, standardize
    // covariances and Cholesky decompositions
    // function scaled back to the original scale
  f = (
    global_offset + 
    rep_matrix(state_offset, N)' + 
    rep_matrix(covariate_f, N)' + 
    rep_matrix(L_global * z_global, D) + 
    L_f * z_f * k_f
  );
    for(n in 1:total_obs) {
      log_lik[n] = normal_lpdf(
        to_vector(y)[control_idx[n]] |
        to_vector(f)[control_idx[n]],
        sigman * sqrt(to_vector(inv_population)[control_idx[n]])
      );
    }
    for(i in 1:N){
      f_samples[i] = to_row_vector(normal_rng(
          f[i],
          sigman * sqrt(inv_population[i])
        ));
    }
}
