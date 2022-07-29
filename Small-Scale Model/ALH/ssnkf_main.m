%%

%Small-Scale New Keynesian Model

clear
clc

%% Characteristics of model

number_endogenous_variables = 3; %Number of endogenous variables
number_exogenous_variables = 3; %Number of exogenous variables
number_aux_variables = 1; %Number of auxiliary variables
number_jumper_variables = 2; %Number of jumper variables
number_observed_variables = 3; %Number of observable variables

%Number of variables in state transition equation in Sims' canonical form
number_state_variables_sims = number_endogenous_variables + number_exogenous_variables + number_aux_variables + number_jumper_variables;

%Number of variables in state transition equation
number_state_variables = number_endogenous_variables + number_exogenous_variables + number_aux_variables;

%% Load the parameter estimates

parameter_info = load('ssnkf_alh_parameter_estimates.csv');

%% Number of periods to generate the IRFs
T = 20;

%Impulse responses
resp = zeros(number_state_variables,number_exogenous_variables,T+2);

%% Parameters

%Fixed parameters
parrhomp = 0;

%Estimated parameters
parsigmag = parameter_info(1);
parsigmaz = parameter_info(2);
parsigmaR = parameter_info(3);
partau = parameter_info(4);
parkappa = parameter_info(5);
parrhoR = parameter_info(6);
parpsi1 = parameter_info(7);
parpsi2 = parameter_info(8);
parrhog = parameter_info(9);
parrhoz = parameter_info(10);
parrA = parameter_info(11);
parpiA = parameter_info(12);
pargammaQ = parameter_info(13);
gna = parameter_info(14);
gnb = parameter_info(15);
omegaa = parameter_info(16);

%Composite parameters
parbeta = 1/(1+(parrA/400));

theta = [parsigmag;parsigmaz;parsigmaR;partau;parkappa;parrhoR;parpsi1;parpsi2;parrhog;parrhoz;parrA;parpiA;pargammaQ;gna;gnb;omegaa];

%% Put the model into canonical form

[RE_Gamma_0,RE_Gamma_1,RE_Gamma_c,RE_Psi,RE_Pi] = ssnkf_ree_canonical_form(number_exogenous_variables,number_jumper_variables,number_state_variables_sims,theta);

%% Get the rational expectations solution using Sims' method

[Phi_1_sims,Phi_c_sims,Phi_epsilon_sims,P1_bar_sims,P2_bar_sims,solution_sims,Phi_epsilon_transform_sims] = ree_sims(number_endogenous_variables,number_exogenous_variables,RE_Gamma_0,RE_Gamma_1,RE_Gamma_c,RE_Psi,RE_Pi);

%% Put the model into condensed form

[G_1,G_2,G_3,G_4,H_1,H_2,invertible,Sigma_v] = ssnkf_alh_condensed_form(number_endogenous_variables,number_exogenous_variables,RE_Gamma_0,RE_Gamma_1,theta);

%% Get the rational expectations solution using Uhlig's method

[Phi_1_uhlig,Phi_epsilon_uhlig,P1_bar_uhlig,P2_bar_uhlig] = ree_uhlig(G_1./omegaa,-eye(number_endogenous_variables),G_3,zeros(number_endogenous_variables,number_exogenous_variables),G_4,H_1,H_2);

%% Find the heterogeneous expectations equilibrium

[hee] = hee_solution(number_endogenous_variables,number_exogenous_variables,G_1,G_2,G_3,G_4,H_1,Sigma_v);

%% Build the adaptive learning state-space matrices that aren't a function of agent beliefs

[K_1,K_2,K_3,K_4,Sigma_epsilon,t,Psi_0,Psi_1,Psi_2,S_c] = ssnkf_build_state_space_matrices(number_endogenous_variables,number_exogenous_variables,number_aux_variables,number_observed_variables,G_1,G_2,G_3,G_4,H_1,H_2,theta);

%% Initialize all variables

%Initialize the adaptive learning algorithm
[s,S_1,S_2,A,B,R_A,R_B,z_A,z_B,error_count] = initialize_learning(number_endogenous_variables,number_exogenous_variables,number_state_variables,T+2,hee);

%Generate the shocks (a one standard deviation shock in the first period (i.e., period 3), zero values for the shocks in all subsequent periods
epsilon = zeros(number_exogenous_variables,T+2);
epsilon(:,3) = ones(number_exogenous_variables,1);

% Store the initial values of the S_1 and S_2 matrices by updating the adaptive learning state-space matrices that are functions of agent beliefs with the given values of the parameters and the initial values of the agent beliefs
[S_1(:,:,2),S_2(:,:,2),~,A(:,:,2),B(:,:,2),R_A(:,:,2),R_B(:,:,2)] = alh_update_state_space_matrices(number_endogenous_variables,number_exogenous_variables,number_aux_variables,K_1,K_2,K_3,K_4,H_1,H_2,A(:,:,2),B(:,:,2),R_A(:,:,2),R_B(:,:,2),hee,A(:,:,1),B(:,:,1),R_A(:,:,1),R_B(:,:,1),S_1(:,:,1),S_2(:,:,1));

%% RE IRFs
size_resp_rows = size(Phi_epsilon_uhlig,1);
size_resp_columns = number_exogenous_variables;
resp_re = zeros(size_resp_rows,size_resp_columns,T+2);
for j = 3:T+2
    if j == 3
        resp_re(:,:,j) = Phi_epsilon_uhlig; % Stores the i-th response of the variables to the shocks.
    else
        resp_re(:,:,j) = Phi_1_uhlig*resp_re(:,:,j-1);
    end
end

%Exogenous spending shock
resp_g_envy_re(:,1) = (squeeze(resp_re(1,1,:)));  
resp_g_envpi_re(:,1) = (squeeze(resp_re(2,1,:)));  
resp_g_envR_re(:,1) = (squeeze(resp_re(3,1,:)));  

%Technology shock
resp_z_envy_re(:,1) = (squeeze(resp_re(1,2,:)));  
resp_z_envpi_re(:,1) = (squeeze(resp_re(2,2,:)));  
resp_z_envR_re(:,1) = (squeeze(resp_re(3,2,:)));

%Monetary policy shock
resp_mp_envy_re(:,1) = (squeeze(resp_re(1,3,:)));  
resp_mp_envpi_re(:,1) = (squeeze(resp_re(2,3,:)));  
resp_mp_envR_re(:,1) = (squeeze(resp_re(3,3,:)));

%% Generate the IRFs by simulating the model

for j = 3:T+2
    
    if j == 3
    
    %Get the last two periods of the state variables
    s_last_two_periods = [s(:,j-2),s(:,j-1)];
    
    %Now update the agent beliefs and moment matrices (the dependent variables are last period's state variables)
    [A(:,:,j),B(:,:,j),R_A(:,:,j),R_B(:,:,j)] = ssnkf_alh_update_beliefs(number_endogenous_variables,number_exogenous_variables,theta,s_last_two_periods,A(:,:,j-1),B(:,:,j-1),R_A(:,:,j-1),R_B(:,:,j-1));
    
    %Now use the updated beliefs to update the adaptive learning state-space matrices that are functions of agent beliefs
    [S_1(:,:,j),S_2(:,:,j),test,A(:,:,j),B(:,:,j),R_A(:,:,j),R_B(:,:,j)] = alh_update_state_space_matrices(number_endogenous_variables,number_exogenous_variables,number_aux_variables,K_1,K_2,K_3,K_4,H_1,H_2,A(:,:,j),B(:,:,j),R_A(:,:,j),R_B(:,:,j),hee,A(:,:,j-1),B(:,:,j-1),R_A(:,:,j-1),R_B(:,:,j-1),S_1(:,:,j-1),S_2(:,:,j-1));

    %Now update the values of the state variables to reflect the updated S_1 and S_2 matrices and the one standard deviation shock to all variables
    s(:,j) = S_1(:,:,j)*s(:,j-1) + S_2(:,:,j)*epsilon(:,j);
    
    %Store the response of the variables to the shocks
    resp(:,:,j) = S_2(:,:,j);
    
    %Record the errors for this iteration
    error_count(j,:) = test;
        
    else
        
    %Get the last two periods of the state variables
    s_last_two_periods = [s(:,j-2),s(:,j-1)];
    
    %Now update the agent beliefs and moment matrices (the dependent variables are last period's state variables)
    [A(:,:,j),B(:,:,j),R_A(:,:,j),R_B(:,:,j)] = ssnkf_alh_update_beliefs(number_endogenous_variables,number_exogenous_variables,theta,s_last_two_periods,A(:,:,j-1),B(:,:,j-1),R_A(:,:,j-1),R_B(:,:,j-1));
    
    %Now use the updated beliefs to update the adaptive learning state-space matrices that are functions of agent beliefs
    [S_1(:,:,j),S_2(:,:,j),test,A(:,:,j),B(:,:,j),R_A(:,:,j),R_B(:,:,j)] = alh_update_state_space_matrices(number_endogenous_variables,number_exogenous_variables,number_aux_variables,K_1,K_2,K_3,K_4,H_1,H_2,A(:,:,j),B(:,:,j),R_A(:,:,j),R_B(:,:,j),hee,A(:,:,j-1),B(:,:,j-1),R_A(:,:,j-1),R_B(:,:,j-1),S_1(:,:,j-1),S_2(:,:,j-1));
    
    %Now update the values of the state variables to reflect the updated S_1 matrix and the fact that the shocks go to zero
    s(:,j) = S_1(:,:,j)*s(:,j-1) + S_2(:,:,j)*epsilon(:,j);
    
    %Store the response of the variables to the shocks
    resp(:,:,j) = S_1(:,:,j)*resp(:,:,j-1);
    
    %Record the errors for this iteration
    error_count(j,:) = test;
    
    end
    
end

%Exogenous spending shock
resp_g_envy(:,1) = (squeeze(resp(1,1,:)));  
resp_g_envpi(:,1) = (squeeze(resp(2,1,:)));  
resp_g_envR(:,1) = (squeeze(resp(3,1,:)));  

%Technology shock
resp_z_envy(:,1) = (squeeze(resp(1,2,:)));  
resp_z_envpi(:,1) = (squeeze(resp(2,2,:)));  
resp_z_envR(:,1) = (squeeze(resp(3,2,:)));

%Monetary policy shock
resp_mp_envy(:,1) = (squeeze(resp(1,3,:)));  
resp_mp_envpi(:,1) = (squeeze(resp(2,3,:)));  
resp_mp_envR(:,1) = (squeeze(resp(3,3,:)));
