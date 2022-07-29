%%
%Small-Scale New Keynesian Model with Heterogeneous Expectations

%Forecasting file

%This algorithm comes from the following 2 sources:
%1. "DSGE Model-Based Forecasting" by Del Negro and Schorfheide, pages 15-16
%2. "Forecasting with DSGE Models" by Christoffel, Coenen, and Warne, page 18

clear
clc

%% Set seed state
seed = 123;
rng(seed);

%% Characteristics of model

number_endogenous_variables = 3; %Number of endogenous variables
number_exogenous_variables = 3; %Number of exogenous variables
number_aux_variables = 1; %Number of auxiliary variables
number_jumper_variables = 2; %Number of jumper variables
number_observed_variables = 3; %Number of observable variables

%% Get different variables numbers based on the model characteristics

%Number of variables in state transition equation in Sims' canonical form
number_state_variables_sims = number_endogenous_variables + number_exogenous_variables + number_aux_variables + number_jumper_variables;

%Number of variables in state transition equation
number_state_variables = number_endogenous_variables + number_exogenous_variables + number_aux_variables; 

%Number of exogenous plus endogenous variables
number_total_variables = number_endogenous_variables + number_exogenous_variables;

%% Load data set

%Data set
%1 = 2001
%2 = 2007
%3 = 2020
data_set_identifier = 1;

[mh_theta,mh_theta_A,mh_theta_B,mh_theta_R_A,mh_theta_R_B,mh_theta_s,mh_theta_P,data_forecast_sample,T,H,TpH,Sigma_u_sd] = ssnkf_alh_load_data_set_forecast(data_set_identifier);

%% Simulation characteristics

%Number of simulations to run for each value of theta
m1 = 1;

%Number of "thetas" to use
m2 = 100000;

%Total number of simulations run
m_total = m1*m2;

%Credible interval proportion
percentile = 0.1;
upper_bound = 100-0.5*(100*percentile);
lower_bound = 0.5*(100*percentile);

%% Create storage for summary statistics

%Storage for forecasts
forecasts = zeros(H,number_observed_variables,m1,m2);

%Storage for simulation errors
errors = zeros(m1,2,m2);

%% Start the "sampling the future" algorithm

for count_m2 = 1:m2
    
    %1. Draw theta
    
    %Stack the parameters into a column vector
    theta = mh_theta(count_m2,:)';
    
    %Parameters
    parrA = theta(11);
    parpiA = theta(12);
    pargammaQ = theta(13);
       
    %2. Get a draw of the state variables at time T using the Kalman filter
    
    % Get the matrices of Sims' canonical form
    [RE_Gamma_0,RE_Gamma_1,~,~,~] = ssnkf_ree_canonical_form(number_exogenous_variables,number_jumper_variables,number_state_variables_sims,theta);
    
    % Get the matrices of the condensed form
    [G_1,G_2,G_3,G_4,H_1,H_2,invertible,Sigma_v] = ssnkf_alh_condensed_form(number_endogenous_variables,number_exogenous_variables,RE_Gamma_0,RE_Gamma_1,theta);
    
    %Find the HEE
    [hee] = hee_solution(number_endogenous_variables,number_exogenous_variables,G_1,G_2,G_3,G_4,H_1,Sigma_v);
    
    %Initialize the adaptive learning algorithm
    [s,S_1,S_2,A,B,R_A,R_B,~,~,error_count] = initialize_learning_forecasting(number_endogenous_variables,number_exogenous_variables,number_state_variables,TpH,mh_theta_A(:,:,count_m2),mh_theta_B(:,:,count_m2),mh_theta_R_A(:,:,count_m2),mh_theta_R_B(:,:,count_m2));
    
    %Build the adaptive learning state-space matrices that aren't a function of agent beliefs
    [K_1,K_2,K_3,K_4,Sigma_epsilon,t,Psi_0,Psi_1,Psi_2,S_c] = ssnkf_build_state_space_matrices(number_endogenous_variables,number_exogenous_variables,number_aux_variables,number_observed_variables,G_1,G_2,G_3,G_4,H_1,H_2,theta);

    %Store the initial values of the S_1 and S_2 matrices by updating the adaptive learning state-space matrices that are functions of agent beliefs with the given values of the parameters and the initial values of the agent beliefs
    [S_1(:,:,2),S_2(:,:,2),test_initial,A(:,:,2),B(:,:,2),R_A(:,:,2),R_B(:,:,2)] = alh_update_state_space_matrices(number_endogenous_variables,number_exogenous_variables,number_aux_variables,K_1,K_2,K_3,K_4,H_1,H_2,A(:,:,2),B(:,:,2),R_A(:,:,2),R_B(:,:,2),hee,A(:,:,1),B(:,:,1),R_A(:,:,1),R_B(:,:,1),S_1(:,:,1),S_2(:,:,1));

    %Get the filter estimate and the covariance matrix of the state variables
    s_filter_estimate = mh_theta_s(:,2,count_m2);
    P_filter_estimate = nearestSPD(mh_theta_P(:,:,count_m2));%The "nearestSPD" function ensures that the matrix P will be symmetric and postive semi-definite so that it can be used in the "mvnrnd" function
        
    for count_m1 = 1:m1

        %Generate a draw of the state variables at time T
        s(:,1) = mvnrnd(s_filter_estimate,P_filter_estimate)';
        s(:,2) = mvnrnd(s_filter_estimate,P_filter_estimate)';

        %3. Simulate a path of the state variables by using the draw of state variables as the initial values and generating a sequence of shocks

        %Initialize storage for individual state variables of interest
        envy = zeros(1,TpH);
        envpi = zeros(1,TpH);
        envR = zeros(1,TpH);
        exvz = zeros(1,TpH);

        %Initialize storage for individual observational equivalent variables
        model_obsgry = zeros(1,TpH);
        model_obspi = zeros(1,TpH);
        model_obsR = zeros(1,TpH);

        %Generate the shocks
        epsilon = normrnd(0,1,number_exogenous_variables,TpH);

        %Simulate the model
        for j = 3:TpH   

            %Get the last two periods of the state variables
            s_last_two_periods = [s(:,j-2),s(:,j-1)];

            %Now update the agent beliefs (the dependent variables are last period's state variables)
            [A(:,:,j),B(:,:,j),R_A(:,:,j),R_B(:,:,j)] = ssnkf_alh_update_beliefs(number_endogenous_variables,number_exogenous_variables,theta,s_last_two_periods,A(:,:,j-1),B(:,:,j-1),R_A(:,:,j-1),R_B(:,:,j-1));

            %Now use the updated beliefs to update the adaptive learning state-space matrices that are functions of agent beliefs
            [S_1(:,:,j),S_2(:,:,j),test,A(:,:,j),B(:,:,j),R_A(:,:,j),R_B(:,:,j)] = alh_update_state_space_matrices(number_endogenous_variables,number_exogenous_variables,number_aux_variables,K_1,K_2,K_3,K_4,H_1,H_2,A(:,:,j),B(:,:,j),R_A(:,:,j),R_B(:,:,j),hee,A(:,:,j-1),B(:,:,j-1),R_A(:,:,j-1),R_B(:,:,j-1),S_1(:,:,j-1),S_2(:,:,j-1));
            
            %Now update the values of the state variables to reflect the updated S_1 and S_2 matrices
            s(:,j) = S_1(:,:,j)*s(:,j-1) + S_2(:,:,j)*epsilon(:,j);

            %Get relevant variables
            envy(j) = s(1,j); %Output growth
            envpi(j) = s(2,j); %Inflation
            envR(j) = s(3,j); %Nominal interest rate
            exvz(j) = s(5,j); %Technology shock process

            %Calculate observational equivalent variables
            model_obsgry(j) = pargammaQ + 100*envy(j) - 100*envy(j-1) + 100*exvz(j);
            model_obspi(j) = parpiA + 400*envpi(j);
            model_obsR(j) = parpiA + parrA + 4*pargammaQ + 400*envR(j);
            
            %Record the errors for this iteration
            error_count(j,:) = test;

        end

        %Drop the extraneous values
        model_obsgry = (model_obsgry(3:end))'; %Output growth
        model_obspi = (model_obspi(3:end))'; %Inflation
        model_obsR = (model_obsR(3:end))'; %Nominal interest rate

        forecasts(:,:,count_m1,count_m2) = [model_obsgry,model_obspi,model_obsR];
        
        %Errors
        errors(count_m1,:,count_m2) = sum(error_count(T+1:TpH,:));

    end

end

%Point forecasts
forecasts_obsgry = reshape(forecasts(:,1,:,:),H,m1*m2);
forecasts_obspi = reshape(forecasts(:,2,:,:),H,m1*m2);
forecasts_obsR = reshape(forecasts(:,3,:,:),H,m1*m2);

forecasts_point_obsgry = mean(forecasts_obsgry,2);
forecasts_point_obspi = mean(forecasts_obspi,2);
forecasts_point_obsR = mean(forecasts_obsR,2);

forecasts_errors_squared_dobsgry = (data_forecast_sample(:,1) - forecasts_point_obsgry).^2;
forecasts_errors_squared_obspi = (data_forecast_sample(:,2) - forecasts_point_obspi).^2;
forecasts_errors_squared_obsR = (data_forecast_sample(:,3) - forecasts_point_obsR).^2;

rsfe.obsgry = sqrt(forecasts_errors_squared_dobsgry);
rsfe.obspi = sqrt(forecasts_errors_squared_obspi);
rsfe.obsR = sqrt(forecasts_errors_squared_obsR);

%Interval forecasts
forecasts_interval_obsgry = [prctile(forecasts_obsgry,lower_bound,2),prctile(forecasts_obsgry,upper_bound,2)];
forecasts_interval_obspi = [prctile(forecasts_obspi,lower_bound,2),prctile(forecasts_obspi,upper_bound,2)];
forecasts_interval_obsR = [prctile(forecasts_obsR,lower_bound,2),prctile(forecasts_obsR,upper_bound,2)];

forecasts_interval_obsgry_length = forecasts_interval_obsgry(:,2) - forecasts_interval_obsgry(:,1);
forecasts_interval_obspi_length = forecasts_interval_obspi(:,2) - forecasts_interval_obspi(:,1);
forecasts_interval_obsR_length = forecasts_interval_obsR(:,2) - forecasts_interval_obsR(:,1);

