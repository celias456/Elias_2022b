function [K_1,K_2,K_3,K_4,Sigma_epsilon,t,Psi_0,Psi_1,Psi_2,S_c] = msnkf_build_state_space_matrices(number_endogenous_variables,number_exogenous_variables,number_aux_variables,number_observed_variables,G_1,G_2,G_3,G_4,H_1,H_2,theta)
%Mediume-Scale New Keynesian Model

%This function builds the state-space matrices that aren't a function of
%agent beliefs

%Input:
%number_endogenous_variables: number of endogenous variables
%number_exogenous_variables: number of exogenous variables
%number_aux_variables: number of auxiliary variables
%number_observed_variables: number of observed variables
%number_state_variables: number of state variables
%theta: vector of parameters

%Output:
%K_1: matrix in condensed form equation
%K_2: matrix in condensed form equation
%K_3: matrix in condensed form equation
%K_4: matrix in condensed form equation
%Sigma_epsilon: sigma_epsilon matrix
%t: vector representing trend values in the measurement equation
%Psi_0: matrix in measurement equation
%Psi_1: matrix in measurement equation
%Psi_2: matrix in measurement equation
%S_c: state equation constants

%Zero matrices
Zero_m = zeros(number_exogenous_variables);
Zero_k = zeros(number_aux_variables);
Zero_nm = zeros(number_endogenous_variables,number_exogenous_variables);
Zero_mn = zeros(number_exogenous_variables,number_endogenous_variables);
Zero_nk = zeros(number_endogenous_variables,number_aux_variables);
Zero_kn = zeros(number_aux_variables,number_endogenous_variables);
Zero_mk = zeros(number_exogenous_variables,number_aux_variables);
Zero_km = zeros(number_aux_variables,number_exogenous_variables);

%Build the auxiliary matrix
Aux_3 = zeros(number_aux_variables,number_endogenous_variables);

Aux_3(1,19) = 1;
Aux_3(2,18) = 1;
Aux_3(3,16) = 1;
Aux_3(4,22) = 1;

%Build the K matrices
K_1 = [G_1,Zero_nm,Zero_nk;Zero_mn,Zero_m,Zero_mk;Zero_kn,Zero_km,Zero_k];
K_2 = [G_2,Zero_nm,Zero_nk;Zero_mn,Zero_m,Zero_mk;Zero_kn,Zero_km,Zero_k];
K_3 = [G_3,G_4*H_1,Zero_nk;Zero_mn,H_1,Zero_mk;Aux_3,Zero_km,Zero_k];
K_4 = [G_4*H_2;H_2;Zero_km];

%Now build the Sigma_epsilon matrix (i.e., the covariance matrix for the stochastic shocks)
Sigma_epsilon = eye(number_exogenous_variables);

%% Now build the measurement equation matrices

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
 
%Vector representing trend values in the measurement equation
t = ones(number_observed_variables,1);

%Initialize Psi matrices
Psi_0 = zeros(number_observed_variables,1);
Psi_1 = zeros(number_observed_variables,number_observed_variables);
Psi_2 = zeros(number_observed_variables,number_endogenous_variables+number_exogenous_variables+number_aux_variables);

%Equation 1
Psi_0(1,1) = ctrend;

Psi_2(1,19) = 1;
Psi_2(1,32) = -1;

%Equation 2
Psi_0(2,1) = ctrend;

Psi_2(2,18) = 1;
Psi_2(2,33) = -1;

%Equation 3
Psi_0(3,1) = ctrend;

Psi_2(3,16) = 1;
Psi_2(3,34) = -1;

%Equation 4
Psi_0(4,1) = ctrend;

Psi_2(4,22) = 1;
Psi_2(4,35) = -1;

%Equation 5
Psi_0(5,1) = constelab;

Psi_2(5,20) = 1;

%Equation 6
Psi_0(6,1) = constepinf;

Psi_2(6,21) = 1;

%Equation 7
Psi_0(7,1) = conster;

Psi_2(7,23) = 1;

%% Now build the state equation constants

S_c = zeros(number_endogenous_variables+number_exogenous_variables+number_aux_variables,1);

end

