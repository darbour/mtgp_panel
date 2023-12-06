data {
  int<lower=1> N;      // number of observations
  int<lower=1> D;      // number of units
  vector[N] x;         // univariate covariate
  matrix[N, D] y;         // target variable
  matrix[N, D] inv_population;
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
  
  real<lower=0> sigman;
  vector[N] z_global;
  
  row_vector[D] state_offset;
}
model {
  matrix[N, N] K_global = gp_exp_quad_cov(xn, sigma_global, lengthscale_global);
  matrix[N, N] L_global = cholesky_decompose(add_diag(K_global, jitter));
  // priors
  z_global ~ std_normal();
  lengthscale_global ~ inv_gamma(5,5);
  sigma_global ~ normal(0, 0.5);
  sigman ~ normal(0, 1);
  state_offset ~ normal(0, 1);

  // index in to only consider the likelihood of the units under control
  to_vector(y)[control_idx] ~ normal(
    to_vector(rep_matrix(state_offset, N) + rep_matrix(L_global * z_global, D))[control_idx],
     + sigman * sqrt(to_vector(inv_population))[control_idx]
  );
}
generated quantities {
  matrix[N, D] f;
  int total_obs = N * D - num_treated;
  vector[N * D - num_treated] log_lik;
  matrix[N, D] f_samples;
  {
    // covariances and Cholesky decompositions
    matrix[N, N] K_global = gp_exp_quad_cov(xn, sigma_global, lengthscale_global);
    matrix[N, N] L_global = cholesky_decompose(add_diag(K_global, jitter));
    // function scaled back to the original scale
    f = rep_matrix(state_offset, N) + rep_matrix(L_global * z_global, D);
    for(n in 1:total_obs) {
      log_lik[n] = normal_lpdf(to_vector(y)[control_idx[n]] | to_vector(f)[control_idx[n]], sigman * sqrt(to_vector(inv_population)[control_idx[n]]));
    }
    for(i in 1:N){
      f_samples[i] = to_row_vector(normal_rng(
          f[i],
          sigman * sqrt(inv_population[i])
        ));
    }
  }
  // real sigma = sigman*ysd;
}
