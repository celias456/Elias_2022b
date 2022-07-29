%%
%Medium-Scale New Keynesian Model with Homogeneous Expectations

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

number_endogenous_variables = 24; %Number of endogenous variables
number_exogenous_variables = 7; %Number of exogenous variables
number_aux_variables = 4; %Number of auxiliary variables
number_jumper_variables = 12; %Number of jumper variables
number_observed_variables = 7; %Number of observable variables

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
data_set_identifier = 3;

[mh_theta,mh_theta_A,mh_theta_B,mh_theta_R_A,mh_theta_R_B,mh_theta_s,mh_theta_P,data_forecast_sample,T,H,TpH,Sigma_u_sd] = msnkf_al_load_data_set_forecast(data_set_identifier);

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
    csigma = theta(9);
    constepinf = theta(22);
    constebeta = theta(23);
    constelab = theta(24);
    ctrend = theta(25);

    %Composite parameters
    cpie = 1 + (constepinf/100);
    cbeta = 1/(1+constebeta/100);
    cgamma = 1 + ctrend/100;
    cr = cpie/(cbeta*cgamma^(-csigma));
    conster = (cr-1)*100;
       
    %2. Get a draw of the state variables at time T using the Kalman filter
    
    % Get the matrices of Sims' canonical form
    [RE_Gamma_0,RE_Gamma_1,~,~,~] = msnkf_ree_canonical_form(number_exogenous_variables,number_jumper_variables,number_state_variables_sims,theta);
    
    % Get the matrices of the condensed form
    [G_1,G_2,G_3,G_4,H_1,H_2,invertible,Sigma_v] = msnkf_al_condensed_form(number_endogenous_variables,number_exogenous_variables,RE_Gamma_0,RE_Gamma_1,theta);
    
    %Find the HEE
    [hee] = hee_solution(number_endogenous_variables,number_exogenous_variables,G_1,G_2,G_3,G_4,H_1,Sigma_v);
    
    %Initialize the adaptive learning algorithm   
    [s,S_1,S_2,A,B,R_A,R_B,~,~,error_count] = initialize_learning_forecasting(number_endogenous_variables,number_exogenous_variables,number_state_variables,TpH,mh_theta_A(:,:,count_m2),mh_theta_B(:,:,count_m2),mh_theta_R_A(:,:,count_m2),mh_theta_R_B(:,:,count_m2));
    
    %Build the adaptive learning state-space matrices that aren't a function of agent beliefs
    [K_1,K_2,K_3,K_4,Sigma_epsilon,t,Psi_0,Psi_1,Psi_2,S_c] = msnkf_build_state_space_matrices(number_endogenous_variables,number_exogenous_variables,number_aux_variables,number_observed_variables,G_1,G_2,G_3,G_4,H_1,H_2,theta);

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
        y = zeros(1,TpH);
        c = zeros(1,TpH);
        inve = zeros(1,TpH);
        w = zeros(1,TpH);
        lab = zeros(1,TpH);
        pinf = zeros(1,TpH);
        r = zeros(1,TpH);

        %Initialize storage for individual observational equivalent variables
        model_dy = zeros(1,TpH);
        model_dc = zeros(1,TpH);
        model_dinve = zeros(1,TpH);
        model_dw = zeros(1,TpH);
        model_labobs = zeros(1,TpH);
        model_pinfobs = zeros(1,TpH);
        model_robs = zeros(1,TpH);

        %Generate the shocks
        epsilon = normrnd(0,1,number_exogenous_variables,TpH);

        %Simulate the model
        for j = (3):(TpH)   

            %Get the last two periods of the state variables
            s_last_two_periods = [s(:,j-2),s(:,j-1)];

            %Now update the agent beliefs (the dependent variables are last period's state variables)
            [A(:,:,j),B(:,:,j),R_A(:,:,j),R_B(:,:,j)] = msnkf_al_update_beliefs(number_endogenous_variables,number_exogenous_variables,theta,s_last_two_periods,A(:,:,j-1),B(:,:,j-1),R_A(:,:,j-1),R_B(:,:,j-1));

            %Now use the updated beliefs to update the adaptive learning state-space matrices that are functions of agent beliefs
            [S_1(:,:,j),S_2(:,:,j),test,A(:,:,j),B(:,:,j),R_A(:,:,j),R_B(:,:,j)] = alh_update_state_space_matrices(number_endogenous_variables,number_exogenous_variables,number_aux_variables,K_1,K_2,K_3,K_4,H_1,H_2,A(:,:,j),B(:,:,j),R_A(:,:,j),R_B(:,:,j),hee,A(:,:,j-1),B(:,:,j-1),R_A(:,:,j-1),R_B(:,:,j-1),S_1(:,:,j-1),S_2(:,:,j-1));
            
            %Now update the values of the state variables to reflect the updated S_1 and S_2 matrices
            s(:,j) = S_1(:,:,j)*s(:,j-1) + S_2(:,:,j)*epsilon(:,j);

            %Get relevant variables
            y(j) = s(19,j); %Output
            c(j) = s(18,j); %Consumption
            inve(j) = s(16,j); %Investment
            w(j) = s(22,j); %Real wage
            lab(j) = s(20,j); %Hours worked
            pinf(j) = s(21,j); %Inflation
            r(j) = s(23,j); %Nominal interest rate

            %Calculate observational equivalent variables
            model_dy(j) = y(j) - y(j-1) + ctrend;
            model_dc(j) = c(j) - c(j-1) + ctrend;
            model_dinve(j) = inve(j) - inve(j-1) + ctrend;
            model_dw(j) = w(j) - w(j-1) + ctrend;
            model_labobs(j) = lab(j) + constelab;
            model_pinfobs(j) = pinf(j) + constepinf;
            model_robs(j) = r(j) + conster;
            
            %Record the errors for this iteration
            error_count(j,:) = test;

        end

        %Drop the extraneous values
        model_dy = (model_dy(3:end))'; %Output growth
        model_dc = (model_dc(3:end))'; %Consumption growth
        model_dinve = (model_dinve(3:end))'; %Investment growth
        model_dw = (model_dw(3:end))'; %Real wage growth
        model_labobs = (model_labobs(3:end))'; %Labor hours
        model_pinfobs = (model_pinfobs(3:end))'; %Inflation
        model_robs = (model_robs(3:end))'; %Nominal interest rate

        forecasts(:,:,count_m1,count_m2) = [model_dy,model_dc,model_dinve,model_dw,model_labobs,model_pinfobs,model_robs];
        
        %Errors
        errors(count_m1,:,count_m2) = sum(error_count(T+1:TpH,:));

    end

end

%Point forecasts
forecasts_dy = reshape(forecasts(:,1,:,:),H,m1*m2);
forecasts_dc = reshape(forecasts(:,2,:,:),H,m1*m2);
forecasts_dinve = reshape(forecasts(:,3,:,:),H,m1*m2);
forecasts_dw = reshape(forecasts(:,4,:,:),H,m1*m2);
forecasts_labobs = reshape(forecasts(:,5,:,:),H,m1*m2);
forecasts_pinfobs = reshape(forecasts(:,6,:,:),H,m1*m2);
forecasts_robs = reshape(forecasts(:,7,:,:),H,m1*m2);

forecasts_point_dy = mean(forecasts_dy,2);
forecasts_point_dc = mean(forecasts_dc,2);
forecasts_point_dinve = mean(forecasts_dinve,2);
forecasts_point_dw = mean(forecasts_dw,2);
forecasts_point_labobs = mean(forecasts_labobs,2);
forecasts_point_pinfobs = mean(forecasts_pinfobs,2);
forecasts_point_robs = mean(forecasts_robs,2);

forecasts_errors_squared_dy = (data_forecast_sample(:,1) - forecasts_point_dy).^2;
forecasts_errors_squared_dc = (data_forecast_sample(:,2) - forecasts_point_dc).^2;
forecasts_errors_squared_dinve = (data_forecast_sample(:,3) - forecasts_point_dinve).^2;
forecasts_errors_squared_dw = (data_forecast_sample(:,4) - forecasts_point_dw).^2;
forecasts_errors_squared_labobs = (data_forecast_sample(:,5) - forecasts_point_labobs).^2;
forecasts_errors_squared_pinfobs = (data_forecast_sample(:,6) - forecasts_point_pinfobs).^2;
forecasts_errors_squared_robs = (data_forecast_sample(:,7) - forecasts_point_robs).^2;

rsfe.dy = sqrt(forecasts_errors_squared_dy);
rsfe.dc = sqrt(forecasts_errors_squared_dc);
rsfe.dinve = sqrt(forecasts_errors_squared_dinve);
rsfe.dw = sqrt(forecasts_errors_squared_dw);
rsfe.labobs = sqrt(forecasts_errors_squared_labobs);
rsfe.pinfobs = sqrt(forecasts_errors_squared_pinfobs);
rsfe.robs = sqrt(forecasts_errors_squared_robs);

%Interval forecasts
forecasts_interval_dy = [prctile(forecasts_dy,lower_bound,2),prctile(forecasts_dy,upper_bound,2)];
forecasts_interval_dc = [prctile(forecasts_dc,lower_bound,2),prctile(forecasts_dc,upper_bound,2)];
forecasts_interval_dinve = [prctile(forecasts_dinve,lower_bound,2),prctile(forecasts_dinve,upper_bound,2)];
forecasts_interval_dw = [prctile(forecasts_dw,lower_bound,2),prctile(forecasts_dw,upper_bound,2)];
forecasts_interval_labobs = [prctile(forecasts_labobs,lower_bound,2),prctile(forecasts_labobs,upper_bound,2)];
forecasts_interval_pinfobs = [prctile(forecasts_pinfobs,lower_bound,2),prctile(forecasts_pinfobs,upper_bound,2)];
forecasts_interval_robs = [prctile(forecasts_robs,lower_bound,2),prctile(forecasts_robs,upper_bound,2)];

forecasts_interval_dy_length = forecasts_interval_dy(:,2) - forecasts_interval_dy(:,1);
forecasts_interval_dc_length = forecasts_interval_dc(:,2) - forecasts_interval_dc(:,1);
forecasts_interval_dinve_length = forecasts_interval_dinve(:,2) - forecasts_interval_dinve(:,1);
forecasts_interval_dw_length = forecasts_interval_dw(:,2) - forecasts_interval_dw(:,1);
forecasts_interval_labobs_length = forecasts_interval_labobs(:,2) - forecasts_interval_labobs(:,1);
forecasts_interval_pinfobs_length = forecasts_interval_pinfobs(:,2) - forecasts_interval_pinfobs(:,1);
forecasts_interval_robs_length = forecasts_interval_robs(:,2) - forecasts_interval_robs(:,1);
