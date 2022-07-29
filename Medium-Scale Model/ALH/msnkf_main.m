%%

%Medium-Scale New Keynesian Model

clear
clc

%% Characteristics of model

number_endogenous_variables = 24; %Number of endogenous variables
number_exogenous_variables = 7; %Number of exogenous variables
number_aux_variables = 4; %Number of auxiliary variables
number_jumper_variables = 12; %Number of jumper variables
number_observed_variables = 7; %Number of observable variables

%Number of variables in state transition equation in Sims' canonical form
number_state_variables_sims = number_endogenous_variables + number_exogenous_variables + number_aux_variables + number_jumper_variables;

%Number of variables in state transition equation
number_state_variables = number_endogenous_variables + number_exogenous_variables + number_aux_variables;

%% Load the parameter estimates

parameter_info = load('msnkf_alh_parameter_estimates.csv');

%% Number of periods to generate the IRFs
T = 20;

%Impulse responses
resp = zeros(number_state_variables,number_exogenous_variables,T+2);

%% Parameters

%Fixed Parameters
ctou = 0.025;
cg = 0.18;
clandaw = 1.5;
curvp = 10;
curvw = 10;

%Parameters
sig_ea = parameter_info(1);
sig_eb = parameter_info(2);
sig_eg = parameter_info(3);
sig_eqs = parameter_info(4);
sig_em = parameter_info(5);
sig_epinf = parameter_info(6);
sig_ew = parameter_info(7);
csadjcost = parameter_info(8);
csigma = parameter_info(9);
chabb = parameter_info(10);
cprobw = parameter_info(11);
csigl = parameter_info(12);
cprobp = parameter_info(13);
cindw = parameter_info(14);
cindp = parameter_info(15);
czcap = parameter_info(16);
cfc = parameter_info(17);
crpi = parameter_info(18);
crr = parameter_info(19);
cry = parameter_info(20);
crdy = parameter_info(21);
constepinf = parameter_info(22);
constebeta = parameter_info(23);
constelab = parameter_info(24);
ctrend = parameter_info(25);
calfa = parameter_info(26);
crhoa = parameter_info(27);
crhob = parameter_info(28);
crhog = parameter_info(29);
crhoqs = parameter_info(30);
crhoms = parameter_info(31);
crhopinf = parameter_info(32);
crhow = parameter_info(33);
gna = parameter_info(34);
gnb = parameter_info(35);
omegaa = parameter_info(36);

%Composite parameters
cpie = 1 + (constepinf/100);
cbeta = 1/(1+constebeta/100);
cgamma = 1 + ctrend/100;
clandap = cfc;
cbetabar = cbeta*cgamma^(-csigma);
cr = cpie/(cbeta*cgamma^(-csigma));
crk = (cbeta^(-1))*(cgamma^csigma) - (1-ctou);
cw = (calfa^calfa*(1-calfa)^(1-calfa)/(clandap*crk^calfa))^(1/(1-calfa));
cikbar = (1-(1-ctou)/cgamma);
cik = (1-(1-ctou)/cgamma)*cgamma;
clk = ((1-calfa)/calfa)*(crk/cw);
cky = cfc*(clk)^(calfa-1);
ciy = cik*cky;
ccy = 1 - cg - cik*cky;
crkky = crk*cky;
cwhlc =  (1/clandaw)*(1-calfa)/calfa*crk*cky/ccy;
conster = (cr-1)*100;
omegab = 1-omegaa;

theta = [sig_ea;sig_eb;sig_eg;sig_eqs;sig_em;sig_epinf;sig_ew;csadjcost;csigma;chabb;cprobw;csigl;cprobp;cindw;cindp;czcap;cfc;crpi;crr;cry;crdy;constepinf;constebeta;constelab;ctrend;calfa;crhoa;crhob;crhog;crhoqs;crhoms;crhopinf;crhow;gna;gnb;omegaa];

%% Put the model into canonical form

[RE_Gamma_0,RE_Gamma_1,RE_Gamma_c,RE_Psi,RE_Pi] = msnkf_ree_canonical_form(number_exogenous_variables,number_jumper_variables,number_state_variables_sims,theta);

%% Get the rational expectations solution using Sims' method

[Phi_1_sims,Phi_c_sims,Phi_epsilon_sims,P1_bar_sims,P2_bar_sims,solution_sims,Phi_epsilon_transform_sims] = ree_sims(number_endogenous_variables,number_exogenous_variables,RE_Gamma_0,RE_Gamma_1,RE_Gamma_c,RE_Psi,RE_Pi);

%% Put the model into condensed form

[G_1,G_2,G_3,G_4,H_1,H_2,invertible,Sigma_v] = msnkf_alh_condensed_form(number_endogenous_variables,number_exogenous_variables,RE_Gamma_0,RE_Gamma_1,theta);

%% Get the rational expectations solution using Uhlig's method

[Phi_1_uhlig,Phi_epsilon_uhlig,P1_bar_uhlig,P2_bar_uhlig] = ree_uhlig(G_1./omegaa,-eye(number_endogenous_variables),G_3,zeros(number_endogenous_variables,number_exogenous_variables),G_4,H_1,H_2);

%% Find the heterogeneous expectations equilibrium

[hee] = hee_solution(number_endogenous_variables,number_exogenous_variables,G_1,G_2,G_3,G_4,H_1,Sigma_v);

%% Build the adaptive learning state-space matrices that aren't a function of agent beliefs

[K_1,K_2,K_3,K_4,Sigma_epsilon,t,Psi_0,Psi_1,Psi_2,S_c] = msnkf_build_state_space_matrices(number_endogenous_variables,number_exogenous_variables,number_aux_variables,number_observed_variables,G_1,G_2,G_3,G_4,H_1,H_2,theta);

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

%Productivity shock
resp_a_mc_re(:,1) = (squeeze(resp_re(12,1,2:end))); %Gross price markup 
resp_a_zcap_re(:,1) = (squeeze(resp_re(13,1,2:end))); %Capital utilization rate 
resp_a_rk_re(:,1) = (squeeze(resp_re(14,1,2:end))); %Rental rate of capital 
resp_a_k_re(:,1) = (squeeze(resp_re(15,1,2:end))); %Capital services 
resp_a_inve_re(:,1) = (squeeze(resp_re(16,1,2:end))); %Investment 
resp_a_pk_re(:,1) = (squeeze(resp_re(17,1,2:end))); %Real value of existing capital stock 
resp_a_c_re(:,1) = (squeeze(resp_re(18,1,2:end))); %Consumption 
resp_a_y_re(:,1) = (squeeze(resp_re(19,1,2:end))); %Output 
resp_a_lab_re(:,1) = (squeeze(resp_re(20,1,2:end))); %Hours worked 
resp_a_pinf_re(:,1) = (squeeze(resp_re(21,1,2:end))); %Inflation 
resp_a_w_re(:,1) = (squeeze(resp_re(22,1,2:end))); %Real wage 
resp_a_r_re(:,1) = (squeeze(resp_re(23,1,2:end))); %Nominal interest rate 
resp_a_kp_re(:,1) = (squeeze(resp_re(24,1,2:end))); %Capital stock 

%Risk premium shock
resp_b_mc_re(:,1) = (squeeze(resp_re(12,2,2:end))); %Gross price markup 
resp_b_zcap_re(:,1) = (squeeze(resp_re(13,2,2:end))); %Capital utilization rate 
resp_b_rk_re(:,1) = (squeeze(resp_re(14,2,2:end))); %Rental rate of capital 
resp_b_k_re(:,1) = (squeeze(resp_re(15,2,2:end))); %Capital services 
resp_b_inve_re(:,1) = (squeeze(resp_re(16,2,2:end))); %Investment 
resp_b_pk_re(:,1) = (squeeze(resp_re(17,2,2:end))); %Real value of existing capital stock 
resp_b_c_re(:,1) = (squeeze(resp_re(18,2,2:end))); %Consumption 
resp_b_y_re(:,1) = (squeeze(resp_re(19,2,2:end))); %Output 
resp_b_lab_re(:,1) = (squeeze(resp_re(20,2,2:end))); %Hours worked 
resp_b_pinf_re(:,1) = (squeeze(resp_re(21,2,2:end))); %Inflation 
resp_b_w_re(:,1) = (squeeze(resp_re(22,2,2:end))); %Real wage 
resp_b_r_re(:,1) = (squeeze(resp_re(23,2,2:end))); %Nominal interest rate 
resp_b_kp_re(:,1) = (squeeze(resp_re(24,2,2:end))); %Capital stock 

%Exogenous spending shock
resp_g_mc_re(:,1) = (squeeze(resp_re(12,3,2:end))); %Gross price markup 
resp_g_zcap_re(:,1) = (squeeze(resp_re(13,3,2:end))); %Capital utilization rate 
resp_g_rk_re(:,1) = (squeeze(resp_re(14,3,2:end))); %Rental rate of capital 
resp_g_k_re(:,1) = (squeeze(resp_re(15,3,2:end))); %Capital services 
resp_g_inve_re(:,1) = (squeeze(resp_re(16,3,2:end))); %Investment 
resp_g_pk_re(:,1) = (squeeze(resp_re(17,3,2:end))); %Real value of existing capital stock 
resp_g_c_re(:,1) = (squeeze(resp_re(18,3,2:end))); %Consumption 
resp_g_y_re(:,1) = (squeeze(resp_re(19,3,2:end))); %Output 
resp_g_lab_re(:,1) = (squeeze(resp_re(20,3,2:end))); %Hours worked 
resp_g_pinf_re(:,1) = (squeeze(resp_re(21,3,2:end))); %Inflation 
resp_g_w_re(:,1) = (squeeze(resp_re(22,3,2:end))); %Real wage 
resp_g_r_re(:,1) = (squeeze(resp_re(23,3,2:end))); %Nominal interest rate 
resp_g_kp_re(:,1) = (squeeze(resp_re(24,3,2:end))); %Capital stock 

%Investment-specific technology shock
resp_qs_mc_re(:,1) = (squeeze(resp_re(12,4,2:end))); %Gross price markup 
resp_qs_zcap_re(:,1) = (squeeze(resp_re(13,4,2:end))); %Capital utilization rate 
resp_qs_rk_re(:,1) = (squeeze(resp_re(14,4,2:end))); %Rental rate of capital 
resp_qs_k_re(:,1) = (squeeze(resp_re(15,4,2:end))); %Capital services 
resp_qs_inve_re(:,1) = (squeeze(resp_re(16,4,2:end))); %Investment 
resp_qs_pk_re(:,1) = (squeeze(resp_re(17,4,2:end))); %Real value of existing capital stock 
resp_qs_c_re(:,1) = (squeeze(resp_re(18,4,2:end))); %Consumption 
resp_qs_y_re(:,1) = (squeeze(resp_re(19,4,2:end))); %Output 
resp_qs_lab_re(:,1) = (squeeze(resp_re(20,4,2:end))); %Hours worked 
resp_qs_pinf_re(:,1) = (squeeze(resp_re(21,4,2:end))); %Inflation 
resp_qs_w_re(:,1) = (squeeze(resp_re(22,4,2:end))); %Real wage 
resp_qs_r_re(:,1) = (squeeze(resp_re(23,4,2:end))); %Nominal interest rate 
resp_qs_kp_re(:,1) = (squeeze(resp_re(24,4,2:end))); %Capital stock 

%Monetary policy shock
resp_ms_mc_re(:,1) = (squeeze(resp_re(12,5,2:end))); %Gross price markup 
resp_ms_zcap_re(:,1) = (squeeze(resp_re(13,5,2:end))); %Capital utilization rate 
resp_ms_rk_re(:,1) = (squeeze(resp_re(14,5,2:end))); %Rental rate of capital 
resp_ms_k_re(:,1) = (squeeze(resp_re(15,5,2:end))); %Capital services 
resp_ms_inve_re(:,1) = (squeeze(resp_re(16,5,2:end))); %Investment 
resp_ms_pk_re(:,1) = (squeeze(resp_re(17,5,2:end))); %Real value of existing capital stock
resp_ms_c_re(:,1) = (squeeze(resp_re(18,5,2:end))); %Consumption 
resp_ms_y_re(:,1) = (squeeze(resp_re(19,5,2:end))); %Output 
resp_ms_lab_re(:,1) = (squeeze(resp_re(20,5,2:end))); %Hours worked 
resp_ms_pinf_re(:,1) = (squeeze(resp_re(21,5,2:end))); %Inflation 
resp_ms_w_re(:,1) = (squeeze(resp_re(22,5,2:end))); %Real wage 
resp_ms_r_re(:,1) = (squeeze(resp_re(23,5,2:end))); %Nominal interest rate 
resp_ms_kp_re(:,1) = (squeeze(resp_re(24,5,2:end))); %Capital stock

%Price markup shock
resp_spinf_mc_re(:,1) = (squeeze(resp_re(12,6,2:end))); %Gross price markup 
resp_spinf_zcap_re(:,1) = (squeeze(resp_re(13,6,2:end))); %Capital utilization rate 
resp_spinf_rk_re(:,1) = (squeeze(resp_re(14,6,2:end))); %Rental rate of capital 
resp_spinf_k_re(:,1) = (squeeze(resp_re(15,6,2:end))); %Capital services 
resp_spinf_inve_re(:,1) = (squeeze(resp_re(16,6,2:end))); %Investment 
resp_spinf_pk_re(:,1) = (squeeze(resp_re(17,6,2:end))); %Real value of existing capital stock 
resp_spinf_c_re(:,1) = (squeeze(resp_re(18,6,2:end))); %Consumption 
resp_spinf_y_re(:,1) = (squeeze(resp_re(19,6,2:end))); %Output 
resp_spinf_lab_re(:,1) = (squeeze(resp_re(20,6,2:end))); %Hours worked 
resp_spinf_pinf_re(:,1) = (squeeze(resp_re(21,6,2:end))); %Inflation 
resp_spinf_w_re(:,1) = (squeeze(resp_re(22,6,2:end))); %Real wage 
resp_spinf_r_re(:,1) = (squeeze(resp_re(23,6,2:end))); %Nominal interest rate 
resp_spinf_kp_re(:,1) = (squeeze(resp_re(24,6,2:end))); %Capital stock 

%Wage markup shock
resp_sw_mc_re(:,1) = (squeeze(resp_re(12,7,2:end))); %Gross price markup 
resp_sw_zcap_re(:,1) = (squeeze(resp_re(13,7,2:end))); %Capital utilization rate 
resp_sw_rk_re(:,1) = (squeeze(resp_re(14,7,2:end))); %Rental rate of capital 
resp_sw_k_re(:,1) = (squeeze(resp_re(15,7,2:end))); %Capital services 
resp_sw_inve_re(:,1) = (squeeze(resp_re(16,7,2:end))); %Investment 
resp_sw_pk_re(:,1) = (squeeze(resp_re(17,7,2:end))); %Real value of existing capital stock 
resp_sw_c_re(:,1) = (squeeze(resp_re(18,7,2:end))); %Consumption 
resp_sw_y_re(:,1) = (squeeze(resp_re(19,7,2:end))); %Output 
resp_sw_lab_re(:,1) = (squeeze(resp_re(20,7,2:end))); %Hours worked 
resp_sw_pinf_re(:,1) = (squeeze(resp_re(21,7,2:end))); %Inflation 
resp_sw_w_re(:,1) = (squeeze(resp_re(22,7,2:end))); %Real wage 
resp_sw_r_re(:,1) = (squeeze(resp_re(23,7,2:end))); %Nominal interest rate 
resp_sw_kp_re(:,1) = (squeeze(resp_re(24,7,2:end))); %Capital stock 

%% Generate the IRFs by simulating the model

for j = 3:T+2
    
    if j == 3
    
    %Get the last two periods of the state variables
    s_last_two_periods = [s(:,j-2),s(:,j-1)];
    
    %Now update the agent beliefs and moment matrices (the dependent variables are last period's state variables)
    [A(:,:,j),B(:,:,j),R_A(:,:,j),R_B(:,:,j)] = msnkf_alh_update_beliefs(number_endogenous_variables,number_exogenous_variables,theta,s_last_two_periods,A(:,:,j-1),B(:,:,j-1),R_A(:,:,j-1),R_B(:,:,j-1));
    
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
    [A(:,:,j),B(:,:,j),R_A(:,:,j),R_B(:,:,j)] = msnkf_alh_update_beliefs(number_endogenous_variables,number_exogenous_variables,theta,s_last_two_periods,A(:,:,j-1),B(:,:,j-1),R_A(:,:,j-1),R_B(:,:,j-1));
    
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

%Productivity shock
resp_a_mc(:,1) = (squeeze(resp(12,1,2:end))); %Gross price markup 
resp_a_zcap(:,1) = (squeeze(resp(13,1,2:end))); %Capital utilization rate 
resp_a_rk(:,1) = (squeeze(resp(14,1,2:end))); %Rental rate of capital 
resp_a_k(:,1) = (squeeze(resp(15,1,2:end))); %Capital services 
resp_a_inve(:,1) = (squeeze(resp(16,1,2:end))); %Investment 
resp_a_pk(:,1) = (squeeze(resp(17,1,2:end))); %Real value of existing capital stock 
resp_a_c(:,1) = (squeeze(resp(18,1,2:end))); %Consumption 
resp_a_y(:,1) = (squeeze(resp(19,1,2:end))); %Output 
resp_a_lab(:,1) = (squeeze(resp(20,1,2:end))); %Hours worked 
resp_a_pinf(:,1) = (squeeze(resp(21,1,2:end))); %Inflation 
resp_a_w(:,1) = (squeeze(resp(22,1,2:end))); %Real wage 
resp_a_r(:,1) = (squeeze(resp(23,1,2:end))); %Nominal interest rate 
resp_a_kp(:,1) = (squeeze(resp(24,1,2:end))); %Capital stock 

%Risk premium shock
resp_b_mc(:,1) = (squeeze(resp(12,2,2:end))); %Gross price markup 
resp_b_zcap(:,1) = (squeeze(resp(13,2,2:end))); %Capital utilization rate 
resp_b_rk(:,1) = (squeeze(resp(14,2,2:end))); %Rental rate of capital 
resp_b_k(:,1) = (squeeze(resp(15,2,2:end))); %Capital services 
resp_b_inve(:,1) = (squeeze(resp(16,2,2:end))); %Investment 
resp_b_pk(:,1) = (squeeze(resp(17,2,2:end))); %Real value of existing capital stock 
resp_b_c(:,1) = (squeeze(resp(18,2,2:end))); %Consumption 
resp_b_y(:,1) = (squeeze(resp(19,2,2:end))); %Output 
resp_b_lab(:,1) = (squeeze(resp(20,2,2:end))); %Hours worked 
resp_b_pinf(:,1) = (squeeze(resp(21,2,2:end))); %Inflation 
resp_b_w(:,1) = (squeeze(resp(22,2,2:end))); %Real wage 
resp_b_r(:,1) = (squeeze(resp(23,2,2:end))); %Nominal interest rate 
resp_b_kp(:,1) = (squeeze(resp(24,2,2:end))); %Capital stock 

%Exogenous spending shock
resp_g_mc(:,1) = (squeeze(resp(12,3,2:end))); %Gross price markup 
resp_g_zcap(:,1) = (squeeze(resp(13,3,2:end))); %Capital utilization rate 
resp_g_rk(:,1) = (squeeze(resp(14,3,2:end))); %Rental rate of capital 
resp_g_k(:,1) = (squeeze(resp(15,3,2:end))); %Capital services 
resp_g_inve(:,1) = (squeeze(resp(16,3,2:end))); %Investment 
resp_g_pk(:,1) = (squeeze(resp(17,3,2:end))); %Real value of existing capital stock 
resp_g_c(:,1) = (squeeze(resp(18,3,2:end))); %Consumption 
resp_g_y(:,1) = (squeeze(resp(19,3,2:end))); %Output 
resp_g_lab(:,1) = (squeeze(resp(20,3,2:end))); %Hours worked 
resp_g_pinf(:,1) = (squeeze(resp(21,3,2:end))); %Inflation 
resp_g_w(:,1) = (squeeze(resp(22,3,2:end))); %Real wage 
resp_g_r(:,1) = (squeeze(resp(23,3,2:end))); %Nominal interest rate 
resp_g_kp(:,1) = (squeeze(resp(24,3,2:end))); %Capital stock 

%Investment-specific technology shock
resp_qs_mc(:,1) = (squeeze(resp(12,4,2:end))); %Gross price markup 
resp_qs_zcap(:,1) = (squeeze(resp(13,4,2:end))); %Capital utilization rate 
resp_qs_rk(:,1) = (squeeze(resp(14,4,2:end))); %Rental rate of capital 
resp_qs_k(:,1) = (squeeze(resp(15,4,2:end))); %Capital services 
resp_qs_inve(:,1) = (squeeze(resp(16,4,2:end))); %Investment 
resp_qs_pk(:,1) = (squeeze(resp(17,4,2:end))); %Real value of existing capital stock 
resp_qs_c(:,1) = (squeeze(resp(18,4,2:end))); %Consumption 
resp_qs_y(:,1) = (squeeze(resp(19,4,2:end))); %Output 
resp_qs_lab(:,1) = (squeeze(resp(20,4,2:end))); %Hours worked 
resp_qs_pinf(:,1) = (squeeze(resp(21,4,2:end))); %Inflation 
resp_qs_w(:,1) = (squeeze(resp(22,4,2:end))); %Real wage 
resp_qs_r(:,1) = (squeeze(resp(23,4,2:end))); %Nominal interest rate 
resp_qs_kp(:,1) = (squeeze(resp(24,4,2:end))); %Capital stock 

%Monetary policy shock
resp_ms_mc(:,1) = (squeeze(resp(12,5,2:end))); %Gross price markup 
resp_ms_zcap(:,1) = (squeeze(resp(13,5,2:end))); %Capital utilization rate 
resp_ms_rk(:,1) = (squeeze(resp(14,5,2:end))); %Rental rate of capital 
resp_ms_k(:,1) = (squeeze(resp(15,5,2:end))); %Capital services 
resp_ms_inve(:,1) = (squeeze(resp(16,5,2:end))); %Investment 
resp_ms_pk(:,1) = (squeeze(resp(17,5,2:end))); %Real value of existing capital stock
resp_ms_c(:,1) = (squeeze(resp(18,5,2:end))); %Consumption 
resp_ms_y(:,1) = (squeeze(resp(19,5,2:end))); %Output 
resp_ms_lab(:,1) = (squeeze(resp(20,5,2:end))); %Hours worked 
resp_ms_pinf(:,1) = (squeeze(resp(21,5,2:end))); %Inflation 
resp_ms_w(:,1) = (squeeze(resp(22,5,2:end))); %Real wage 
resp_ms_r(:,1) = (squeeze(resp(23,5,2:end))); %Nominal interest rate 
resp_ms_kp(:,1) = (squeeze(resp(24,5,2:end))); %Capital stock

%Price markup shock
resp_spinf_mc(:,1) = (squeeze(resp(12,6,2:end))); %Gross price markup 
resp_spinf_zcap(:,1) = (squeeze(resp(13,6,2:end))); %Capital utilization rate 
resp_spinf_rk(:,1) = (squeeze(resp(14,6,2:end))); %Rental rate of capital 
resp_spinf_k(:,1) = (squeeze(resp(15,6,2:end))); %Capital services 
resp_spinf_inve(:,1) = (squeeze(resp(16,6,2:end))); %Investment 
resp_spinf_pk(:,1) = (squeeze(resp(17,6,2:end))); %Real value of existing capital stock 
resp_spinf_c(:,1) = (squeeze(resp(18,6,2:end))); %Consumption 
resp_spinf_y(:,1) = (squeeze(resp(19,6,2:end))); %Output 
resp_spinf_lab(:,1) = (squeeze(resp(20,6,2:end))); %Hours worked 
resp_spinf_pinf(:,1) = (squeeze(resp(21,6,2:end))); %Inflation 
resp_spinf_w(:,1) = (squeeze(resp(22,6,2:end))); %Real wage 
resp_spinf_r(:,1) = (squeeze(resp(23,6,2:end))); %Nominal interest rate 
resp_spinf_kp(:,1) = (squeeze(resp(24,6,2:end))); %Capital stock 

%Wage markup shock
resp_sw_mc(:,1) = (squeeze(resp(12,7,2:end))); %Gross price markup 
resp_sw_zcap(:,1) = (squeeze(resp(13,7,2:end))); %Capital utilization rate 
resp_sw_rk(:,1) = (squeeze(resp(14,7,2:end))); %Rental rate of capital 
resp_sw_k(:,1) = (squeeze(resp(15,7,2:end))); %Capital services 
resp_sw_inve(:,1) = (squeeze(resp(16,7,2:end))); %Investment 
resp_sw_pk(:,1) = (squeeze(resp(17,7,2:end))); %Real value of existing capital stock 
resp_sw_c(:,1) = (squeeze(resp(18,7,2:end))); %Consumption 
resp_sw_y(:,1) = (squeeze(resp(19,7,2:end))); %Output 
resp_sw_lab(:,1) = (squeeze(resp(20,7,2:end))); %Hours worked 
resp_sw_pinf(:,1) = (squeeze(resp(21,7,2:end))); %Inflation 
resp_sw_w(:,1) = (squeeze(resp(22,7,2:end))); %Real wage 
resp_sw_r(:,1) = (squeeze(resp(23,7,2:end))); %Nominal interest rate 
resp_sw_kp(:,1) = (squeeze(resp(24,7,2:end))); %Capital stock 