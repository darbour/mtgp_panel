data {
  int<lower=1> N;      // number of observations
  int<lower=1> D;      // number of units
  int<lower=1> num_outcomes;
  int<lower=1> n_k_f;      // number of latent functions for f
  vector[N] x;         // univariate covariate
  matrix[N, D] y[num_outcomes];         // target variable
  matrix[N, D] inv_population;
  int num_treated;
  int control_idx[N * D - num_treated];
}
transformed data {
  //matrix[N, D] y_rate = y .* inv_population;
  // Normalize data
  real xmean = mean(x);
  real xsd = sd(x);
  real xn[N] = to_array_1d((x - xmean)/xsd);
  real sigma_intercept = 0.1;
  vector[N] jitter = rep_vector(1e-9, N);
}
parameters {
  real<lower=0> lengthscale_global;
  real<lower=0> sigma_global[num_outcomes];
  real<lower=0> lengthscale_f; // lengthscale of f
  real<lower=0> sigma_f[num_outcomes];       // scale of f
  real<lower=0> sigman[num_outcomes];
  vector[D] state_offset[num_outcomes];
  vector[N] z_global[num_outcomes];
  matrix[N, n_k_f] z_f;
  matrix[n_k_f, D] k_f[num_outcomes];
  real global_offset[num_outcomes];
}

model {
  // covariances and Cholesky decompositions
  matrix[N, N] K_f = gp_exp_quad_cov(xn, 1, lengthscale_f);
  matrix[N, N] L_f = cholesky_decompose(add_diag(K_f, jitter));

  matrix[N, N] K_global = gp_exp_quad_cov(xn, 1.0, lengthscale_global);
  matrix[N, N] L_global = cholesky_decompose(add_diag(K_global, jitter));


  // priors
  to_vector(z_f) ~ std_normal();
  lengthscale_f ~ inv_gamma(5,5);
  lengthscale_global ~ inv_gamma(5,5);
  sigma_f ~ normal(0, 1);
  sigma_global ~ normal(0, 1);
  sigman ~ normal(0, 1);
  global_offset ~ normal(0, 5);

  for(i in 1:num_outcomes) {
    z_global[i] ~ std_normal();
    state_offset[i] ~ normal(0, 5);
    to_vector(k_f[i]) ~ std_normal();
    // index in to only consider the likelihood of the units under control
    to_vector(y[i])[control_idx] ~ normal(
      to_vector(
        global_offset[i] +
        rep_matrix(state_offset[i], N)' +
        rep_matrix(L_global * z_global[i], D) * sigma_global[i]+
        L_f * z_f * k_f[i] * sigma_f[i]
      )[control_idx],
      sigman[i] * sqrt(to_vector(inv_population))[control_idx]
    );
  }
}
generated quantities {
  matrix[N, D] f[num_outcomes];
  matrix[N, D] f_samples[num_outcomes];
  int total_obs = N * D - num_treated;
  vector[N * D - num_treated] log_lik;

  matrix[N, N] K_f = gp_exp_quad_cov(xn, 1.0, lengthscale_f);
  matrix[N, N] L_f = cholesky_decompose(add_diag(K_f, jitter));

  matrix[N, N] K_global = gp_exp_quad_cov(xn, 1.0, lengthscale_global);
  matrix[N, N] L_global = cholesky_decompose(add_diag(K_global, jitter));

  for(j in 1:num_outcomes) {
    f[j] = (
      global_offset[j] + 
      rep_matrix(state_offset[j], N)' + 
      rep_matrix(L_global * z_global[j], D) * sigma_global[j] + 
      L_f * z_f * k_f[j] * sigma_f[j]
    );

    for(n in 1:total_obs) {
      log_lik[n] = normal_lpdf(
        to_vector(y[j])[control_idx[n]] |
        to_vector(f[j])[control_idx[n]],
        sigman[j] * sqrt(to_vector(inv_population)[control_idx[n]])
      );
    }
    for(i in 1:N){
      f_samples[j][i] = to_row_vector(normal_rng(
          f[j][i],
          sigman[j] * sqrt(inv_population[i])
        ));
    }
  }
}
