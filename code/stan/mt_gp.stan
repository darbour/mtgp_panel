data {
  int<lower=1> N;      // number of observations
  int<lower=1> D;      // number of units
  int<lower=1> n_k_f;      // number of latent functions for f
  int<lower=1> n_k_g;    // number of latent functions for the noise
  vector[N] x;         // univariate covariate
  matrix[N, D] y;         // target variable
  matrix[N, D] population;
  int num_treated;
  int control_idx[N * D - num_treated];
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
  real<lower=0> lengthscale_f; // lengthscale of f
  real<lower=0> sigma_f;       // scale of f
  real<lower=0> lengthscale_g; // lengthscale of g
  real<lower=0> sigma_g;       // scale of g
  matrix[N, n_k_f] z_f;
  matrix[n_k_f, D] k_f;
  matrix[N, n_k_g] z_g;
  matrix[n_k_g, D] k_g;

  real<lower=0> lengthscale_global;
  real<lower=0> sigma_global;
  vector[N] z_global;
  row_vector[D] state_offset;
  row_vector[D] state_offset_g;
  
  real intercept;
}
model {
  // covariances and Cholesky decompositions
  matrix[N, N] K_f = gp_exp_quad_cov(xn, sigma_f, lengthscale_f);
  matrix[N, N] L_f = cholesky_decompose(add_diag(K_f, jitter));
  matrix[N, N] K_g = gp_exp_quad_cov(xn, sigma_g, lengthscale_g);
  matrix[N, N] L_g = cholesky_decompose(add_diag(K_g, jitter));

  matrix[N, N] K_global = gp_exp_quad_cov(xn, sigma_global, lengthscale_global);
  matrix[N, N] L_global = cholesky_decompose(add_diag(K_global, jitter));

  lengthscale_global ~ inv_gamma(5,5); // uniform(0,1);//lognormal(log(.3), .2);
  sigma_global ~ std_normal();
  z_global ~ std_normal();
  state_offset ~ std_normal();
  state_offset_g ~ std_normal();

  // priors
  to_vector(z_f) ~ std_normal();
  to_vector(z_g) ~ std_normal();
  to_vector(k_f) ~ std_normal();
  to_vector(k_g) ~ std_normal();
  lengthscale_f ~ inv_gamma(5,5);
  lengthscale_g ~ inv_gamma(5,5);
  sigma_f ~ normal(0, .5);
  sigma_g ~ normal(0, .5);

  // index in to only consider the likelihood of the units under control
  to_vector(y)[control_idx] ~ lognormal(
    intercept + to_vector(log(population) + rep_matrix(state_offset, N) + rep_matrix(L_global * z_global, D) + L_f * z_f * k_f)[control_idx],
    (exp(to_vector(rep_matrix(state_offset_g, N) + L_g * z_g * k_g)) ./ sqrt(to_vector(population)))[control_idx] 
  );
}
generated quantities {
  matrix[N, D] f;
  matrix[N, D] sigma;
  int total_obs = N * D - num_treated;
  vector[N * D - num_treated] log_lik;
  {
    // covariances and Cholesky decompositions
    matrix[N, N] K_f = gp_exp_quad_cov(xn, sigma_f, lengthscale_f);
    matrix[N, N] L_f = cholesky_decompose(add_diag(K_f, jitter));
    matrix[N, N] K_g = gp_exp_quad_cov(xn, sigma_g, lengthscale_g);
    matrix[N, N] L_g = cholesky_decompose(add_diag(K_g, jitter));
    matrix[N, N] K_global = gp_exp_quad_cov(xn, sigma_global, lengthscale_global);
    matrix[N, N] L_global = cholesky_decompose(add_diag(K_global, jitter));
    // function scaled back to the original scale
    f = intercept + log(population) + rep_matrix(state_offset, N) + rep_matrix(L_global * z_global, D) + L_f * z_f * k_f;
    sigma = exp(rep_matrix(state_offset_g, N) + L_g * z_g * k_g) ./ sqrt(population);
    for(n in 1:total_obs) {
      log_lik[n] = lognormal_lpdf(to_vector(y)[control_idx[n]] | to_vector(f)[control_idx[n]], to_vector(sigma ./ sqrt(population))[control_idx[n]]);
    }
  }
}
