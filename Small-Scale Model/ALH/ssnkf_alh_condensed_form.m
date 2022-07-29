function [G_1,G_2,G_3,G_4,H_1,H_2,invertible,Sigma_v] = ssnkf_alh_condensed_form(number_endogenous_variables,number_exogenous_variables,Gamma_0,Gamma_1,theta)
%Small-Scale New Keynesian Model with Heterogeneous Expectations

%This function returns the condensed form matrices

%Sims' canonical form:
%Gamma_0*s_t = Gamma_1*s_{t-1} + Gamma_c + Psi*epsilon_t + Pi*eta_t

%Condensed form:
%x_t = G_1*E_t^A x_{t+1} + G_2*E_t^B x_{t+1} + G_3*x_{t-1} + G_4 v_t
%v_t = H_1*v_{t-1} + H_2*epsilon_t

%x_t is an nx1 vector of endogenous variables, v_t is an mx1 vector of exogenous
%variables and epsilon_t is i.i.d.

%% Parameters

%Proportions of agents
omegaa = theta(end);
omegab = 1-omegaa;

%Shock standard deviations
parsigmag = theta(1);
parsigmaz = theta(2);
parsigmaR = theta(3);

shock_sd = [parsigmag,parsigmaz,parsigmaR];

%Exogenous variable autocorrelation coefficients
parrhog = theta(9);
parrhoz = theta(10);
parrhomp = 0;

ev_acor = [parrhog,parrhoz,parrhomp];

%% Initialize matrices

D_1 = zeros(number_endogenous_variables,number_endogenous_variables);
D_2 = zeros(number_endogenous_variables,number_endogenous_variables);
F_1 = zeros(number_exogenous_variables,number_exogenous_variables);
F_2 = zeros(number_exogenous_variables,number_exogenous_variables);

%% Build D Matrices

D_0 = Gamma_0(1:number_endogenous_variables,1:number_endogenous_variables);

D_1(:,1) = omegaa*(-Gamma_0(1:number_endogenous_variables,8)); %Expectation for envy
D_1(:,2) = omegaa*(-Gamma_0(1:number_endogenous_variables,9)); %Expectation for envpi

D_2(:,1) = omegab*(-Gamma_0(1:number_endogenous_variables,8)); %Expectation for envy
D_2(:,2) = omegab*(-Gamma_0(1:number_endogenous_variables,9)); %Expectation for envpi

D_3 = Gamma_1(1:number_endogenous_variables,1:number_endogenous_variables);

D_4 = -Gamma_0(1:number_endogenous_variables,(number_endogenous_variables+1):(number_endogenous_variables+number_exogenous_variables));

%% Build F Matrices

F_0 = eye(number_exogenous_variables);

for index_1 = 1:number_exogenous_variables
    F_1(index_1,index_1) = ev_acor(index_1);
    F_2(index_1,index_1) = shock_sd(index_1);
end

%% Test the D_0 and F_0 matrices for singularity

%Set the values of all variables in case the matrices are not invertible
invertible = 0;
G_1 = zeros(number_endogenous_variables,number_endogenous_variables);
G_2 = zeros(number_endogenous_variables,number_endogenous_variables);
G_3 = zeros(number_endogenous_variables,number_endogenous_variables);
G_4 = zeros(number_endogenous_variables,number_exogenous_variables);
H_1 = zeros(number_exogenous_variables,number_exogenous_variables);
H_2 = zeros(number_exogenous_variables,number_exogenous_variables);

Sigma_v = zeros(number_exogenous_variables,number_exogenous_variables);

%Test the matrices
[test_result_singular_D_0] = test_matrix_singular(D_0);
[test_result_singular_F_0] = test_matrix_singular(F_0);

if test_result_singular_D_0 == 0 && test_result_singular_F_0 == 0 %D_0 and F_0 are invertible; Now calculate the G and H matrices
    invertible = 1;
    G_1 = D_0\D_1;
    G_2 = D_0\D_2;
    G_3 = D_0\D_3;
    G_4 = D_0\D_4;
    H_1 = F_0\F_1;
    H_2 = F_0\F_2;
    
    %Create the Sigma_v matrix
    Sigma_v(1,1) = (parsigmag^2)/(1-parrhog^2);
    Sigma_v(2,2) = (parsigmaz^2)/(1-parrhoz^2);
    Sigma_v(3,3) = (parsigmaR^2)/(1-parrhomp^2);
    
end

end

