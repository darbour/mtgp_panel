library(haven)
library(tidyverse)
library(cmdstanr)
library(posterior)
options(pillar.neg = FALSE, pillar.subtle=FALSE, pillar.sigfig=2)
library(tidyr)
library(dplyr)
library(ggplot2)
library(bayesplot)
theme_set(bayesplot::theme_default(base_family = "sans", base_size=16))
set1 <- RColorBrewer::brewer.pal(7, "Set1")
SEED <- 48927 # set random seed for reproducibility


df <- read_dta('ucrthrough2018.dta')
crimes_of_interest = c('homicide', 'aggravated_assault')
df %>%
    filter(year >= 1996) %>%
    filter(state_abbr != 'DC') %>%
    pivot_longer(names(df)[4:ncol(df)], 'crimetype', values_to='count') %>%
    arrange(crimetype, state_abbr, year) %>%
    filter(crimetype %in% crimes_of_interest) -> crime_df

df %>%
    filter(year >= 1996) %>%
    filter(state_abbr != 'DC') %>%
    select(state_abbr, year, population) %>%
    pivot_wider(
        id_cols=year,
        names_from=state_abbr,
        values_from=population
    ) -> pop_wide_df

X = pop_wide_df$year
populations = as.matrix(pop_wide_df[,2:ncol(pop_wide_df)])
N = nrow(pop_wide_df)
D = ncol(pop_wide_df) - 1
Y = t(matrix(crime_df$count, N * D, length(crimes_of_interest)))

# CA is the treated unit
treatment_idx = which(colnames(pop_wide_df) == 'CA') - 1
# Assuming 2006 is the treatment year
# First treated year is 2007
treatment_time = which(X == 2007)

# construct an indexing matrix for the untreated observations
excluded_indices = matrix(
        1:length(populations), nrow(populations), ncol(populations)
    )[treatment_time:nrow(Y),treatment_idx]
control_indices = (1:length(populations))[-excluded_indices]

filebf0 = 'stan/mt_gp_pois_multioutput.stan'
model_gpbffg <- cmdstan_model(stan_file = filebf0, include_paths = ".")
standata_gpbffg <- list(x=X,
                        y=Y,
                        N=N,
                        D=D,
                        population=populations,
                        n_k_f=15,
                        n_k_d=10,
                        num_outcomes=length(crimes_of_interest),
                        control_idx=control_indices,
                        num_treated=length(excluded_indices)
                        )

fit_gpcovfg <- model_gpbffg$sample(data=standata_gpbffg,
                                  iter_warmup=500, iter_sampling=500,
                                  chains=4, parallel_chains=4, adapt_delta=0.9, max_treedepth = 13)

draws_gpcovf <- fit_gpcovfg$draws()
print(summarise_draws(subset(draws_gpcovf, variable=c('sigma_','lengthscale_', 'f_'),
                       regex=TRUE)))

draws_gpcovf_m <- as_draws_matrix(draws_gpcovf)
# reshape draws to be [years, states]
Ef = matrix(colMeans(subset(draws_gpcovf_m, variable='f_star')), nrow(Y), ncol(Y))
sigma <- sqrt(matrix(colMeans(subset(draws_gpcovf_m, variable='f_star')), nrow(Y), ncol(Y)))
