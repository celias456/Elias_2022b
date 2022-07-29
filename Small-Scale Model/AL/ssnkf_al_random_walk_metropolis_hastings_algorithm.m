function [mh_theta,mh_theta_log_prior,mh_theta_log_likelihood,mh_theta_log_posterior_kernel,acceptance_rate,mh_theta_A,mh_theta_B,mh_theta_s,mh_theta_R_A,mh_theta_R_B,mh_theta_P] = ssnkf_al_random_walk_metropolis_hastings_algorithm(theta,Sigma_hat,c,number_draws,number_endogenous_variables,data,number_exogenous_variables,number_aux_variables,number_jumper_variables,number_observed_variables,number_state_variables,number_state_variables_sims,number_total_variables,prior_information,Sigma_u_sd,burn_proportion,first_observation,T)
%Small-Scale New Keynesian Model

% Inputs
% theta: Column vector of parameters
% Sigma_hat: Covariance matrix used for the variance of the proposal distribution in the MH algorithm (obtained from Dynare)
% c: Tuning parameter
% number_draws: Number of draws to generate in the algorithm
% number_endogenous_variables: Number of endogenous variables
% data: matrix of data
% number_exogenous_variables: Number of exogenous variables
% number_observed_variables: Number of observed variables
% prior_information: Matrix of prior information
% Sigma_u_sd: Standard deviation of measurement error
% burn_proportion: %Proportion of draws to discard in MH chain
% first_observation: First observation to use in the data set
% T: number of observations in the data set

%% Metropolis-Hastings algorithm characteristics

%Number of parameters
number_paramaters = size(theta,1);

mh_mu = zeros(number_paramaters,1); %Mean of the random walk term (eta) in the MH algorithm
mh_Sigma = (c^2)*Sigma_hat; %Variance of the proposal distribution in the MH algorithm

%% Create storage

%Create storage for the components of the log posterior of each draw in the MH chain
mh_theta = zeros(number_draws,number_paramaters); %Accepted draws in the MH chain
mh_theta_log_prior = zeros(number_draws,1); %Log prior of accepted draw in the MH chain
mh_theta_log_likelihood = zeros(number_draws,1); %Log likelihood of accepted draw in the MH chain
mh_theta_log_posterior_kernel = zeros(number_draws,1); %Log posterior kernel of accepted draw in the MH chain
mh_theta_A = zeros(number_endogenous_variables,number_total_variables,number_draws); %Last learning values for agent-type A associated with each specific value of theta
mh_theta_R_A = zeros(number_total_variables,number_total_variables,number_draws); %Last learning moment matrices for agent-type A associated with each specific value of theta
mh_theta_B = zeros(number_endogenous_variables,number_exogenous_variables,number_draws); %Last learning values for agent-type B associated each specific value of theta
mh_theta_R_B = zeros(number_exogenous_variables,number_exogenous_variables,number_draws); %Last learning moment matrices for agent-type B associated with each specific value of theta
mh_theta_s = zeros(number_state_variables,2,number_draws); %Last two values of the state variables associated with each specific value of theta
mh_theta_P = zeros(number_state_variables,number_state_variables,number_draws); %Last value of the covaraince matrix in the Kalman filter associated with each specific value of theta

%Create storage for the components of the log posterior of each of the candidate draws
mh_theta_candidate = zeros(number_draws,number_paramaters); %Candidate values
mh_theta_candidate_log_prior = zeros(number_draws,1); %Log prior of candidate values
mh_theta_candidate_log_likelihood = zeros(number_draws,1); %Log likelihood of candidate values
mh_theta_candidate_log_posterior_kernel = zeros(number_draws,1); %Log posterior kernel of candidate values
mh_theta_candidate_A = zeros(number_endogenous_variables,number_total_variables,number_draws); %Learning values for agent-type A associated with the candidate values of theta
mh_theta_candidate_R_A = zeros(number_total_variables,number_total_variables,number_draws); %Last learning moment matrices for agent-type A associated with each specific candidate value of theta
mh_theta_candidate_B = zeros(number_endogenous_variables,number_exogenous_variables,number_draws); %Learning values for agent-type B associated with the candidate value of theta
mh_theta_candidate_R_B = zeros(number_exogenous_variables,number_exogenous_variables,number_draws); %Learning moment matrices for agent-type B associated with each specific candidate value of theta
mh_theta_candidate_s = zeros(number_state_variables,2,number_draws); %Values of the state variables associated with the candidate value of theta
mh_theta_candidate_P = zeros(number_state_variables,number_state_variables,number_draws); %Last value of the covaraince matrix in the Kalman filter associated with each specific candidate value of theta

%Create storage for calculating acceptance rate
mh_acceptance = zeros(number_draws,1); %1 if candidate is accepted, zero otherwise

%% Initilize the Metropolis-Hastings algorithm

%Get values of components of log posterior kernel with initial parameter values

[log_prior,log_likelihood,log_posterior,~,A_initial,B_initial,s_initial,~,R_A_initial,R_B_initial,P_initial] = ssnkf_al_log_posterior_calculate(number_endogenous_variables,number_exogenous_variables,number_aux_variables,number_jumper_variables,number_observed_variables,number_state_variables,number_state_variables_sims,data,theta,prior_information,Sigma_u_sd,first_observation,T);                                                     

%Set initial values of the components of the log posterior kernel
mh_theta(1,:) = theta;
mh_theta_log_prior(1) = log_prior;
mh_theta_log_likelihood(1) = log_likelihood;
mh_theta_log_posterior_kernel(1) = log_posterior;
mh_theta_A(:,:,1) = A_initial;
mh_theta_R_A(:,:,1) = R_A_initial;
mh_theta_B(:,:,1) = B_initial;
mh_theta_R_B(:,:,1) = R_B_initial;
mh_theta_s(:,:,1) = s_initial;
mh_theta_P(:,:,1) = P_initial;

%% Start the Metropolis-Hastings algorithm

for mh_draws_index = 2:number_draws
    
    %Generate candidate values of parameters
    mh_theta_candidate(mh_draws_index,:) = mh_theta(mh_draws_index-1,:) + mvnrnd(mh_mu,mh_Sigma);
    
    %Get value of log posterior with candidate values
    [log_prior,log_likelihood,log_posterior,~,A_candidate,B_candidate,s_candidate,~,R_A_candidate,R_B_candidate,P_candidate] = ssnkf_al_log_posterior_calculate(number_endogenous_variables,number_exogenous_variables,number_aux_variables,number_jumper_variables,number_observed_variables,number_state_variables,number_state_variables_sims,data,mh_theta_candidate(mh_draws_index,:)',prior_information,Sigma_u_sd,first_observation,T);
    
    %Test for finiteness of log posterior
    if log_posterior == -Inf %Immediately discard these candidates and move to the next step in the chain
       mh_theta(mh_draws_index,:) = mh_theta(mh_draws_index-1,:);
       mh_theta_log_prior(mh_draws_index) = mh_theta_log_prior(mh_draws_index-1);
       mh_theta_log_likelihood(mh_draws_index) = mh_theta_log_likelihood(mh_draws_index-1);
       mh_theta_log_posterior_kernel(mh_draws_index) = mh_theta_log_posterior_kernel(mh_draws_index-1);
       mh_theta_A(:,:,mh_draws_index) = mh_theta_A(:,:,mh_draws_index-1);
       mh_theta_R_A(:,:,mh_draws_index) = mh_theta_R_A(:,:,mh_draws_index-1);
       mh_theta_B(:,:,mh_draws_index) = mh_theta_B(:,:,mh_draws_index-1);
       mh_theta_R_B(:,:,mh_draws_index) = mh_theta_R_B(:,:,mh_draws_index-1);
       mh_theta_s(:,:,mh_draws_index) = mh_theta_s(:,:,mh_draws_index-1);
       mh_theta_P(:,:,mh_draws_index) = mh_theta_P(:,:,mh_draws_index-1);
    else
       %HEE solution is unique and stable and value of log posterior is finite     
       mh_theta_candidate_log_prior(mh_draws_index) = log_prior;
       mh_theta_candidate_log_likelihood(mh_draws_index) = log_likelihood;
       mh_theta_candidate_log_posterior_kernel(mh_draws_index) = log_posterior;
       mh_theta_candidate_A(:,:,mh_draws_index) = A_candidate;
       mh_theta_candidate_R_A(:,:,mh_draws_index) = R_A_candidate;
       mh_theta_candidate_B(:,:,mh_draws_index) = B_candidate;
       mh_theta_candidate_R_B(:,:,mh_draws_index) = R_B_candidate;
       mh_theta_candidate_s(:,:,mh_draws_index) = s_candidate;
       mh_theta_candidate_P(:,:,mh_draws_index) = P_candidate;
       
       %Determine if candidate values of parameters are accepted
       mh_test = exp(mh_theta_candidate_log_posterior_kernel(mh_draws_index) - mh_theta_log_posterior_kernel(mh_draws_index-1));
       mh_alpha = min(mh_test,1);
       mh_U = unifrnd(0,1);
            if mh_U <= mh_alpha %Candidate is accepted
               mh_theta(mh_draws_index,:) = mh_theta_candidate(mh_draws_index,:);
               mh_theta_log_prior(mh_draws_index) = mh_theta_candidate_log_prior(mh_draws_index);
               mh_theta_log_likelihood(mh_draws_index) = mh_theta_candidate_log_likelihood(mh_draws_index);
               mh_theta_log_posterior_kernel(mh_draws_index) = mh_theta_candidate_log_posterior_kernel(mh_draws_index);
               mh_acceptance(mh_draws_index) = 1;
               mh_theta_A(:,:,mh_draws_index) = mh_theta_candidate_A(:,:,mh_draws_index);
               mh_theta_R_A(:,:,mh_draws_index) = mh_theta_candidate_R_A(:,:,mh_draws_index);
               mh_theta_B(:,:,mh_draws_index) = mh_theta_candidate_B(:,:,mh_draws_index);
               mh_theta_R_B(:,:,mh_draws_index) = mh_theta_candidate_R_B(:,:,mh_draws_index);
               mh_theta_s(:,:,mh_draws_index) = mh_theta_candidate_s(:,:,mh_draws_index);
               mh_theta_P(:,:,mh_draws_index) = mh_theta_candidate_P(:,:,mh_draws_index);
            else %Candidate is not accepted
               mh_theta(mh_draws_index,:) = mh_theta(mh_draws_index-1,:);
               mh_theta_log_prior(mh_draws_index) = mh_theta_log_prior(mh_draws_index-1);
               mh_theta_log_likelihood(mh_draws_index) = mh_theta_log_likelihood(mh_draws_index-1);
               mh_theta_log_posterior_kernel(mh_draws_index) = mh_theta_log_posterior_kernel(mh_draws_index-1);
               mh_theta_A(:,:,mh_draws_index) = mh_theta_A(:,:,mh_draws_index-1);
               mh_theta_R_A(:,:,mh_draws_index) = mh_theta_R_A(:,:,mh_draws_index-1);
               mh_theta_B(:,:,mh_draws_index) = mh_theta_B(:,:,mh_draws_index-1);
               mh_theta_R_B(:,:,mh_draws_index) = mh_theta_R_B(:,:,mh_draws_index-1);
               mh_theta_s(:,:,mh_draws_index) = mh_theta_s(:,:,mh_draws_index-1);
               mh_theta_P(:,:,mh_draws_index) = mh_theta_P(:,:,mh_draws_index-1);
            end
    end
end

%Calculate the acceptance rate
acceptance_rate = (sum(mh_acceptance))/number_draws;

end

