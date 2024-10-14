library(lsirm12pl)

### Parameters 


ndim     <- 2 
niter    <- 55000
nburn    <- 5000
nthin    <- 5
nprint   <- 5000
jump_beta     <- 0.1
jump_theta    <- 0.1
jump_w        <- 0.5
jump_z        <- 0.3
jump_gamma    <- 0.05


pr_mean_beta  <- 0
pr_sd_beta    <- 1
pr_mean_theta <- 0
pr_sd_theta   <- 1
pr_mean_gamma <- 0.0
pr_sd_gamma <- 1.0
pr_a_sigma <- 0.001
pr_b_sigma <- 0.001
pr_a_th_sigma <- 0.001
pr_b_th_sigma <- 0.001

dataset = attention_group_matrix
output_lsirm = lsirm1pl_fixed_gamma(as.matrix(dataset), ndim = 2)

