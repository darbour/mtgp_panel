data {
  int<lower=1> N;      // number of observations
  int<lower=1> D;      // number of units
  int<lower=1> num_outcomes; // number of outcomes we're modeling
  int<lower=1> n_k_f;      // number of latent functions for f
  int<lower=1> n_k_d;      // number of latent functions for units
  vector[N] x;         // univariate covariate
  matrix[N, D] population;
  array[num_outcomes, N * D] int<lower=0> y;         // target variable
  int num_treated;
  array[N * D - num_treated] int control_idx;
}
transformed data {
  // Normalize data
  real xmean = mean(x);
  real xsd = sd(x);
  array[N] real xn = to_array_1d((x - xmean)/xsd);
  vector[N] jitter = rep_vector(1e-9, N);
}
parameters {
  real<lower=0> lengthscale_f; // lengthscale of f
  real<lower=0> sigma_f;       // scale of f

  real<lower=0> lengthscale_global;
  real<lower=0> sigma_global;
  //vector[N] z_global[num_outcomes];
  array[num_outcomes] vector[N] z_global;

  matrix[N, n_k_f] z_f;
  //matrix[n_k_f, n_k_d] k_f[num_outcomes];
  array[num_outcomes] matrix[n_k_f, n_k_d]  k_f;
  matrix[n_k_d, D] k_d;
  //row_vector[D] intercepts[num_outcomes];
  array[num_outcomes] row_vector[D] intercepts;
}
model {
  // covariances and Cholesky decompositions
  matrix[N, N] K_f = gp_exp_quad_cov(xn, sigma_f, lengthscale_f);
  matrix[N, N] L_f = cholesky_decompose(add_diag(K_f, jitter));
  matrix[N, N] K_global = gp_exp_quad_cov(xn, sigma_global, lengthscale_global);
  matrix[N, N] L_global = cholesky_decompose(add_diag(K_global, jitter));
  //matrix[N, D] f[num_outcomes];
  //vector[N] f_global[num_outcomes];
  array[num_outcomes] matrix[N, D] f;
  array[num_outcomes] vector[N] f_global;

  for(i in 1:num_outcomes) {
    f[i] = L_f * z_f * k_f[i] * k_d + rep_matrix(L_global * z_global[i], D);
  }
  // priors
  to_vector(z_f) ~ std_normal();
  for(i in 1:num_outcomes) {
    to_vector(k_f[i]) ~ std_normal();
    z_global[i] ~ std_normal();
    intercepts[i] ~ std_normal();
  }

  //global_cov ~ lkj_corr(1.0);
  lengthscale_f ~ inv_gamma(5, 5); // uniform(0,1);//lognormal(log(.3), .2);
  sigma_f ~ std_normal();
  lengthscale_global ~ inv_gamma(5, 5); // uniform(0,1);//lognormal(log(.3), .2);
  sigma_global ~ std_normal();
  to_vector(k_d) ~ std_normal();

  // index in to only consider the likelihood of the units under control
  for(i in 1:num_outcomes) {
    y[i][control_idx] ~ poisson_log(
      log(to_vector(population)[control_idx]) + to_vector(rep_matrix(intercepts[i], N) + f[i])[control_idx]);
  }
}
generated quantities {
  array[num_outcomes] matrix[N, D] f;
  {
    // covariances and Cholesky decompositions
    matrix[N, N] K_f = gp_exp_quad_cov(xn, sigma_f, lengthscale_f);
    matrix[N, N] L_f = cholesky_decompose(add_diag(K_f, jitter));
    matrix[N, N] K_global = gp_exp_quad_cov(xn, sigma_global, lengthscale_global);
    matrix[N, N] L_global = cholesky_decompose(add_diag(K_global, jitter));
    for(i in 1:num_outcomes) {
      f[i] = exp(log(population) + rep_matrix(intercepts[i], N) + L_f * z_f * k_f[i] * k_d + rep_matrix(L_global * z_global[i], D));
    }
  }
}
