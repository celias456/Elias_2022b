function [G_1,G_2,G_3,G_4,H_1,H_2,invertible,Sigma_v] = msnkf_al_condensed_form(number_endogenous_variables,number_exogenous_variables,Gamma_0,Gamma_1,theta)
%Medium-Scale New Keynesian Model with Homogeneous Expectations

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
omegaa = 1;
omegab = 1-omegaa;

%Shock standard deviations
sig_ea = theta(1);
sig_eb = theta(2);
sig_eg = theta(3);
sig_eqs = theta(4);
sig_em = theta(5);
sig_epinf = theta(6);
sig_ew = theta(7);

shock_sd = [sig_ea,sig_eb,sig_eg,sig_eqs,sig_em,sig_epinf,sig_ew];

%Exogenous variable autocorrelation coefficients
crhoa = theta(27);
crhob = theta(28);
crhog = theta(29);
crhoqs = theta(30);
crhoms = theta(31);
crhopinf = theta(32);
crhow = theta(33);

ev_acor = [crhoa,crhob,crhog,crhoqs,crhoms,crhopinf,crhow];

%% Initialize matrices

D_1 = zeros(number_endogenous_variables,number_endogenous_variables);
D_2 = zeros(number_endogenous_variables,number_endogenous_variables);
F_1 = zeros(number_exogenous_variables,number_exogenous_variables);
F_2 = zeros(number_exogenous_variables,number_exogenous_variables);

%% Build D Matrices

D_0 = Gamma_0(1:number_endogenous_variables,1:number_endogenous_variables);

D_1(:,3) = omegaa*(-Gamma_0(1:number_endogenous_variables,45)); %Expectation for rkf
D_1(:,5) = omegaa*(-Gamma_0(1:number_endogenous_variables,43)); %Expectation for invef
D_1(:,6) = omegaa*(-Gamma_0(1:number_endogenous_variables,44)); %Expectation for pkf
D_1(:,7) = omegaa*(-Gamma_0(1:number_endogenous_variables,46)); %Expectation for cf
D_1(:,9) = omegaa*(-Gamma_0(1:number_endogenous_variables,47)); %Expectation for invef
D_1(:,14) = omegaa*(-Gamma_0(1:number_endogenous_variables,41)); %Expectation for rk
D_1(:,16) = omegaa*(-Gamma_0(1:number_endogenous_variables,39)); %Expectation for inve
D_1(:,17) = omegaa*(-Gamma_0(1:number_endogenous_variables,40)); %Expectation for pk
D_1(:,18) = omegaa*(-Gamma_0(1:number_endogenous_variables,36)); %Expectation for c
D_1(:,20) = omegaa*(-Gamma_0(1:number_endogenous_variables,37)); %Expectation for lab
D_1(:,21) = omegaa*(-Gamma_0(1:number_endogenous_variables,38)); %Expectation for pinf
D_1(:,22) = omegaa*(-Gamma_0(1:number_endogenous_variables,42)); %Expectation for w

D_2(:,3) = omegab*(-Gamma_0(1:number_endogenous_variables,45)); %Expectation for rkf
D_2(:,5) = omegab*(-Gamma_0(1:number_endogenous_variables,43)); %Expectation for invef
D_2(:,6) = omegab*(-Gamma_0(1:number_endogenous_variables,44)); %Expectation for pkf
D_2(:,7) = omegab*(-Gamma_0(1:number_endogenous_variables,46)); %Expectation for cf
D_2(:,9) = omegab*(-Gamma_0(1:number_endogenous_variables,47)); %Expectation for invef
D_2(:,14) = omegab*(-Gamma_0(1:number_endogenous_variables,41)); %Expectation for rk
D_2(:,16) = omegab*(-Gamma_0(1:number_endogenous_variables,39)); %Expectation for inve
D_2(:,17) = omegab*(-Gamma_0(1:number_endogenous_variables,40)); %Expectation for pk
D_2(:,18) = omegab*(-Gamma_0(1:number_endogenous_variables,36)); %Expectation for c
D_2(:,20) = omegab*(-Gamma_0(1:number_endogenous_variables,37)); %Expectation for lab
D_2(:,21) = omegab*(-Gamma_0(1:number_endogenous_variables,38)); %Expectation for pinf
D_2(:,22) = omegab*(-Gamma_0(1:number_endogenous_variables,42)); %Expectation for w

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

if test_result_singular_D_0 == 0 && test_result_singular_F_0 == 0 %D_0 and F_0 are invertible; Now calculate the G, H and Sigma_v matrices
    invertible = 1;
    G_1 = D_0\D_1;
    G_2 = D_0\D_2;
    G_3 = D_0\D_3;
    G_4 = D_0\D_4;
    H_1 = F_0\F_1;
    H_2 = F_0\F_2;
    
    %Create the Sigma_v matrix
    Sigma_v(1,1) = (sig_ea^2)/(1-crhoa^2);
    Sigma_v(2,2) = (sig_eb^2)/(1-crhob^2);
    Sigma_v(3,3) = (sig_eg^2)/(1-crhog^2);
    Sigma_v(4,4) = (sig_eqs^2)/(1-crhoqs^2);
    Sigma_v(5,5) = (sig_em^2)/(1-crhoms^2);
    Sigma_v(6,6) = (sig_epinf^2)/(1-crhopinf^2);
    Sigma_v(7,7) = (sig_ew^2)/(1-crhow^2);
    
end



end

