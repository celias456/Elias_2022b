function [s_bar_initial,P_initial,log_likelihood_values] = kalman_filter_initialize(n,Phi_1,Phi_epsilon,Sigma_epsilon,T)
%Gives initial values for the Kalman Filter
%   State (i.e., plant) equation: 
%   s_t = Phi_1*s_{t-1} + Phi_c + Phi_epsilon*epsilon_t   
%   epsilon_t is i.i.d. N(0,1)                     
%
%   Measurement (i.e., observation) equation: 
%   y_t = Psi_0 + Psi_1*t + Psi_2*s_t + u_t   
%   u_t is distributed as N(0,Sigma_u)                 
%   
%  In the system, y is an (m*1) vector of observable variables and s 
%  is an n*1 vector of state (i.e., latent) variables. There are T observations in the data.
%
%  The outputs of this function are:
%  s_bar_initial: the initial values of the state variables
%  P_initial: the initial covariance matrix of the state variables
%  log_likelihood_value: vector of zeros that will store log-likelihood values at each step in the algorithm

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

s_bar_initial = zeros(n,1); %Initial values of the state variables

%This is the standard way to initialize the P matrix of the Kalman filter
a_1 = (eye(n*n)-kron(Phi_1,Phi_1));
[a_1_inv,~] = matrix_inverse(a_1);
a = a_1_inv*reshape(Phi_epsilon*Sigma_epsilon*Phi_epsilon',n*n,1);

P_initial = reshape(a,n,n); %Initial value of the covariance matrix of the state variables

%This is an alternative way to initialize the P matrix of the Kalman
%filter; I got this from the code from Herbst and Schorfheide book;
%P_initial = nearestSPD(dlyap(Phi_1, Phi_epsilon*Sigma_epsilon*Phi_epsilon'));

log_likelihood_values = zeros(T,1); %Store values for log-likelihood

end

