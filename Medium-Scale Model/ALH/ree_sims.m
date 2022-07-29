function [Phi_1,Phi_c,Phi_epsilon,P1_bar,P2_bar,solution,Phi_epsilon_transform] = ree_sims(number_endogenous_variables,number_exogenous_variables,Gamma_0,Gamma_1,Gamma_c,Psi,Pi)
%This function builds the state space matrices of a linear rational
%expectations model using Chris Sims' gensys

%Canonical form
%Gamma_0*s_t = Gamma_1*s_{t-1} + Gamma_c + Psi*epsilon_t + Pi*eta_t

%Phi_1, Phi_c, and Phi_epsilon are the matrices of the state (plant) equation:
%   s_t = Phi_1*s_{t-1} + Phi_c + Phi_epsilon*epsilon_t

%Sigma_epsilon: Covariance matrix for the stochastic shocks

%or

%  x_t = P1_bar x_{t-1} + P2_bar v_t
%  v_t = H_1 v_{t-1} + H_2 epsilon_t
%
%x_t: vector of endogenous variables
%v_t: vector of exogenous variables
%epsilon_t: vector of i.i.d. disturbances

%Input:
%number_endogenous_variables
%number_exogenous_variables
%Gamma_0
%Gamma_1
%Gamma_c
%Psi
%Pi

%Output:
%Phi_1
%Phi_c
%Phi_epsilon
%P1_bar
%P2_bar
%solution: an indicator of the REE solution; 1 is a unique and stable solution, 0 means the solution is not unique and stable
%Phi_epsilon_transform: Transformed Phi_epsilon matrix (used to match the policy and transition functions in Dynare

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Find the REE solution using Gensys

[Phi_1,Phi_c,Phi_epsilon,~,~,~,~,eu] = gensys(Gamma_0,Gamma_1,Gamma_c,Psi,Pi);

solution_sum = sum(eu);

solution = 0;

if solution_sum == 2
    solution = 1;
end

%% Transform Phi_epsilon matrix to match the Dynare policy and transition function output

number_state_variables = size(Gamma_0,1);

Phi_epsilon_transform = zeros(number_state_variables,number_exogenous_variables);
epsilon_sd = zeros(number_exogenous_variables,1);

for index_1 = 1:number_exogenous_variables
    epsilon_sd(index_1) = Psi(number_endogenous_variables+index_1,index_1);
end

for index_2 = 1:number_exogenous_variables
    Phi_epsilon_transform(:,index_2) = Phi_epsilon(:,index_2)/epsilon_sd(index_2); %Transformed Phi_epsilon matrix
end

%% Find the solution in terms of P1_bar and P2_bar

P1_bar = Phi_1(1:number_endogenous_variables,1:number_endogenous_variables);
P2_bar = Phi_epsilon(1:number_endogenous_variables,1:number_exogenous_variables)/Phi_epsilon((number_endogenous_variables+1):(number_endogenous_variables+number_exogenous_variables),1:number_exogenous_variables);

end

