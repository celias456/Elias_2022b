function [log_prior,log_likelihood,log_posterior,hee,A_last,B_last,s_last_two_periods,error_count,R_A_last,R_B_last,P] = msnkf_alh_log_posterior_calculate(number_endogenous_variables,number_exogenous_variables,number_aux_variables,number_jumper_variables,number_observed_variables,number_state_variables,number_state_variables_sims,data,theta,prior_information,Sigma_u_sd,first_observation,T)
%Medium-Scale New Keynesian Model with Heterogeneous Expectations

%Calculates the value of the log posterior kernel

% Input:
% number_endogenous_variables: number of endogenous variables
% number_exogenous_variables: number of exogenous variables
% number_aux_variables: number of auxiliary variables
% number_jumper_variables: number of jumper variables
% number_observed_variables: number of observed variables
% number_state_variables: number of state variables in state-space system
% number_state_variables_sims: number of state variables in state-space
% system of Sims' canconical form
% data: matrix of data
% theta: column vector of parameters
% prior_information: matrix of prior information
% Sigma_u_sd: standard deviation of measurement error
% first_observation: first observation to use in the data set
% T: number of observations in the data set

% Output:
% log_prior: value of log prior
% log_likelihood: value of log likelihood
% log_posterior: value of log posterior kernel
% hee: structure with four elements:
    %hee.A_1_bar: HEE expressions for the coefficients on agent-type A's endogenous variable regressors
    %hee.A_2_bar: HEE expressions for the coefficients on agent-type A's exogenous variable regressors
    %hee.B_bar: HEE expressions for the coefficients on agent-type B's exogenous variable regressor
    %hee.solution: equals 1 if solution is unique and stable, 0 otherwise
% A: matrix of beliefs for agent-type A
% B: matrix of beliefs for agent-type B
% s: matrix of values of the state variables over the T periods (T is the
    %same size as the number of observations in the data set)
%error_count: counts the number of iterations in which the beliefs had to be set to the initial values
    %1st column is for NaNs or infinity in the S matrix
    %2nd column is for a non-stationary S_1 matrix

%Default values for the components of the log posterior kernel
log_prior = -Inf;
log_likelihood = -Inf;
log_posterior = -Inf;

%Default values for agent beliefs and moment matrices
A_last = 0;
B_last = 0;
R_A_last = 0;
R_B_last = 0;

%Default value for state variables
s_last_two_periods = 0;

%Default value for error count
error_count = 0;

%Default value of hee solution
hee = 0;

%Defalut value of P
P = 0;

%% Get the matrices of Sims' canonical form

[RE_Gamma_0,RE_Gamma_1,~,~,~] = msnkf_ree_canonical_form(number_exogenous_variables,number_jumper_variables,number_state_variables_sims,theta);

%% Get the matrices of the condensed form

[G_1,G_2,G_3,G_4,H_1,H_2,invertible,Sigma_v] = msnkf_alh_condensed_form(number_endogenous_variables,number_exogenous_variables,RE_Gamma_0,RE_Gamma_1,theta);

%% Only find the heterogeneous expectations equilibrium (HEE) if the condensed form matrices were able to be created

if invertible == 1 %The HEE condensed form matrices were able to be created
    
    %Find the HEE
    [hee] = hee_solution(number_endogenous_variables,number_exogenous_variables,G_1,G_2,G_3,G_4,H_1,Sigma_v);

    %Only calculate value of log posterior if the HEE solution is unique and stable
    if hee.solution == 1 %HEE solution is unique and stable

        %Initialize the adaptive learning algorithm
        [s,S_1,S_2,A,B,R_A,R_B,~,~,error_count] = initialize_learning(number_endogenous_variables,number_exogenous_variables,number_state_variables,T,hee);

        %Build the adaptive learning state-space matrices that aren't a function of agent beliefs
        [K_1,K_2,K_3,K_4,Sigma_epsilon,t,Psi_0,Psi_1,Psi_2,S_c] = msnkf_build_state_space_matrices(number_endogenous_variables,number_exogenous_variables,number_aux_variables,number_observed_variables,G_1,G_2,G_3,G_4,H_1,H_2,theta);

        %Store the initial values of the S_1 and S_2 matrices by updating the adaptive learning state-space matrices that are functions of agent beliefs with the given values of the parameters and the initial values of the agent beliefs
        [S_1(:,:,2),S_2(:,:,2),test_initial,A(:,:,2),B(:,:,2),R_A(:,:,2),R_B(:,:,2)] = alh_update_state_space_matrices(number_endogenous_variables,number_exogenous_variables,number_aux_variables,K_1,K_2,K_3,K_4,H_1,H_2,A(:,:,2),B(:,:,2),R_A(:,:,2),R_B(:,:,2),hee,A(:,:,1),B(:,:,1),R_A(:,:,1),R_B(:,:,1),S_1(:,:,1),S_2(:,:,1));

        %Test the initial S_1 matrix for NaN/Inf and non-stationarity
        if test_initial(1) == 0 && test_initial(2) == 0

            %Initialize the Kalman filter
            [s(:,2),P_initial,log_likelihood_values] = kalman_filter_initialize(number_state_variables,S_1(:,:,2),S_2(:,:,2),Sigma_epsilon,T);

            %Start Kalman filter algorithm; in this algorithm, the idea is to update the agent beliefs and the state-space matrices (i.e., the S_1, and S_2 matrices) and calculate the value of the log likelihood at each data point
            for period = first_observation:T

                %Get the new filter estimate of the mean and covariance matrix of the state variables for this particular data point
                [s(:,period),P,lik,~] = kalman_filter(data(period,:)',s(:,period-1),P_initial,S_1(:,:,period-1),S_c,S_2(:,:,period-1),Psi_0,Psi_1,Psi_2,t,Sigma_epsilon,Sigma_u_sd);

                %Get the last two periods of the state variables
                s_last_two_periods = [s(:,period-2),s(:,period-1)];

                %Now update the agent beliefs and the moment matrices (the dependent variables are last period's state variables)
                [A(:,:,period),B(:,:,period),R_A(:,:,period),R_B(:,:,period)] = msnkf_alh_update_beliefs(number_endogenous_variables,number_exogenous_variables,theta,s_last_two_periods,A(:,:,period-1),B(:,:,period-1),R_A(:,:,period-1),R_B(:,:,period-1));

                %Now use the updated beliefs to update the adaptive learning state-space matrices that are functions of agent beliefs
                [S_1(:,:,period),S_2(:,:,period),test,A(:,:,period),B(:,:,period),R_A(:,:,period),R_B(:,:,period)] = alh_update_state_space_matrices(number_endogenous_variables,number_exogenous_variables,number_aux_variables,K_1,K_2,K_3,K_4,H_1,H_2,A(:,:,period),B(:,:,period),R_A(:,:,period),R_B(:,:,period),hee,A(:,:,period-1),B(:,:,period-1),R_A(:,:,period-1),R_B(:,:,period-1),S_1(:,:,period-1),S_2(:,:,period-1));

                %Update the MSE matrix for the next iteration
                P_initial = P;

                %Record the errors for this iteration
                error_count(period,:) = test;

                %Test the value of the likelihood for NaN values
                test_like = test_matrix_nan(lik);
                if test_like == 1
                    lik = -Inf;
                end

                %Record the value of the likelihood for this iteration
                log_likelihood_values(period) = lik;

            end

            %Calculate value of log likelihood
            log_likelihood = sum(log_likelihood_values);

            %Calculate value of log prior with the given parameter values
            log_prior = log_prior_calculate(theta,prior_information);

            %Calculate log posterior kernel
            log_posterior = log_prior + log_likelihood;

            %Test the log posterior for NaN
            log_posterior_test = isnan(log_posterior);

            %If log posterior is NaN, set the components of the log posterior equal to -Inf
            if log_posterior_test == 1
                log_prior = -Inf;
                log_likelihood = -Inf;
                log_posterior = -Inf;
                
            end
            
            %Get the last values of A and B
            A_last = A(:,:,end);
            B_last = B(:,:,end);
            
            %Get the last values of R_A and R_B
            R_A_last = R_A(:,:,end);
            R_B_last = R_B(:,:,end);
            
        end

    end
    
end

end

