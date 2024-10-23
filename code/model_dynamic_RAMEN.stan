
data {

  int n_nodes ;
  int N ;
  int sender_id[N] ;
  int receiver_id[N] ;
  int K ; 
  int Y[N] ;
  real past_weight[N] ;
  real past_reci[N] ;
  real past_trans[N] ;
  real past_in_modu[N] ;
  int n_dyads ;
  int dyad_id[N] ;
  int send_receive[N] ;
  
}


parameters {

  real intercept ;
  cholesky_factor_corr[2] corr_nodes ; 
  vector<lower=0>[2] sigma_nodes ;
  matrix[2, n_nodes] z_nodes ;
  
  cholesky_factor_corr[2] corr_dyads; 
  real<lower=0> sigma_dyads; 
  matrix[2, n_dyads] z_dyads ; 
  
  cholesky_factor_corr[K*2] corr_multi_effects ;
  vector<lower=0>[K*2] sigma_multi_effects ;
  matrix[K * 2, n_nodes] z_multi_effects ;
  
  
  vector[N] effect_adj ;
  real alpha ;
  real b_weight ;
  real b_reci ;
  real b_trans ;
  real b_in_modu ;
  real<lower=0> sigma_effects ;

}

transformed parameters{
  
  matrix[n_nodes, 2] mean_nodes ;
  matrix[n_dyads,2] mean_dyads;
  matrix[n_nodes, K*2] mean_multi_effects ;
  
  mean_nodes = (diag_pre_multiply(sigma_nodes, corr_nodes) * z_nodes )';
  mean_dyads = (diag_pre_multiply(rep_vector(sigma_dyads, 2), corr_dyads) * z_dyads)';
  mean_multi_effects = (diag_pre_multiply(sigma_multi_effects, corr_multi_effects) * z_multi_effects )'; 
  
}

model {

  intercept ~ normal(0,1) ;
  
  //node terms
  to_vector(z_nodes) ~ normal(0,1) ; 
  corr_nodes ~ lkj_corr_cholesky(5) ;
  sigma_nodes ~ gamma(1,1) ; 

  
  //dyad terms
  to_vector(z_dyads) ~ normal(0,1) ; 
  corr_dyads ~ lkj_corr_cholesky(5) ;
  sigma_dyads ~ gamma(1,1) ;
  
  
  //multi-effect terms
  to_vector(z_multi_effects) ~ normal(0,1) ; 
  corr_multi_effects ~ lkj_corr_cholesky(5) ;
  sigma_multi_effects ~ gamma(1,1) ;
  
  
  //latent effects
  alpha ~ normal(0,1) ;
  b_weight ~ normal(0,1) ; 
  b_reci~ normal(0,1) ;
  b_trans~ normal(0,1) ;
  b_in_modu~ normal(0,1) ;
  sigma_effects ~ normal(1,1) ;
  
  // this acts as a "penalty" on the explanatory terms... regularizing effect
  effect_adj ~ normal(0,1) ; 
  
  for(n in 1:N){

    effect_adj[n] ~ normal(b_weight*past_weight[n] + b_reci*past_reci[n]+ b_trans*past_trans[n], 1) ; 

    Y[n] ~ bernoulli(Phi( intercept  + effect_adj[n] +  mean_nodes[sender_id[n],1] + mean_nodes[receiver_id[n],2] + mean_dyads[dyad_id[n], send_receive[n]] + mean_multi_effects[sender_id[n],1:K] * (mean_multi_effects[receiver_id[n],(K+1):(K*2) ] )' )) ;
    
  }

}

generated quantities {
  
  real Y_sim[N] ;
  vector[N] log_lik;
  
  for(n in 1:N){
    
    real p = Phi(  intercept  + effect_adj[n] +  mean_nodes[sender_id[n],1] + mean_nodes[receiver_id[n],2] + mean_dyads[dyad_id[n], send_receive[n]] + mean_multi_effects[sender_id[n],1:K] * (mean_multi_effects[receiver_id[n],(K+1):(K*2) ] )' ) ;
    Y_sim[n] = bernoulli_rng(p) ;
    log_lik[n] = bernoulli_lpmf(Y[n] | p);
    
  }
  
}
