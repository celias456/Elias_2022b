function [s_bar_t_given_t,P_t_given_t,lik,y_tp1_given_t] = kalman_filter(y,s_bar_tm1_given_tm1,P__tm1_given_tm1,Phi_1,Phi_c,Phi_epsilon,Psi_0,Psi_1,Psi_2,t,Sigma_epsilon,Sigma_u_sd)

%==========================================================================
%                               Kalman Filter                     
%                                                                 
%   This Function implements the Kalman filter for the state space model:
%
%   State (i.e., plant) equation: 
%   s_t = Phi_1*s_{t-1} + Phi_c + Phi_epsilon*epsilon_t   
%   epsilon_t is distributed as N(0,Sigma_epsilon)                     
%
%   Measurement (i.e., observation) equation: 
%   y_t = Psi_0 + Psi_1*t + Psi_2*s_t + u_t   
%   u_t is distributed as N(0,Sigma_u)                 
%   
%  In the system, y is an (m*1) vector of observable variables and s 
%  is an n*1 vector of state (i.e., latent) variables. 
%
%  The inputs of this function are:
%  - y: m*1 vector of observations for y_t
%  - s_bar_tm1_given_tm1: n*1 vector of initial values of the state variables
%  - P__tm1_given_tm1: n*n covariance matrix of initial values of the state
%    variables
%  - Phi_1, Phi_c, Phi_epsilon: State transition matrices (from gensys)
%  - Psi_0, Psi_1, Psi_2: Measurement equation matrices
%  - t: m*1 vector of ones representing trend values in the measurement
%    equation
%  - Sigma_epsilon: k*k matrix representing the covariance matrix of the
%    stochastic shocks
%  - Sigma_u_sd: standard deviation of measurement errors
%  
%    
%  
%  The outputs of this function are:
%  - s_bar_t_given_t: the prediction E[st|Y^(t)]
%  - P_t_given_t: the MSE of s_bar
%  - lik = value of the log likelihood;
%  - y_tp1_given_t = the one step ahead prediction for y;
%
% This algorithm comes from the Herbst and Schorfheide book "Bayesian
% Estimation of DSGE Models", page 23 and the associated matlab files that
% were distributed with the book
%
%==========================================================================

%Number of observeables
m = size(y,1);

%Measurement error matrix
Sigma_u = zeros(m,m);
for index_1 = 1:m
    Sigma_u(index_1,index_1) = (Sigma_u_sd*y(index_1))^2;
end

%Psi_2 prime
Psi_2_prime = Psi_2';

% Updating Step
s_bar_t_given_tm1 = Phi_1*s_bar_tm1_given_tm1 + Phi_c;
P_t_given_tm1 = Phi_1*P__tm1_given_tm1*Phi_1' + Phi_epsilon*Sigma_epsilon*Phi_epsilon';

% Prediction Step
y_bar_t_given_tm1 = Psi_0 + Psi_1*t + Psi_2*s_bar_t_given_tm1;
F_t_given_tm1 = Psi_2*P_t_given_tm1*Psi_2_prime + Sigma_u;

%This section is to make sure F_t_given_tm1 is invertible and has a
%positive value for the determinant; A positive value for the determinant
%enusres that the value of the log likelihood won't contain any imaginary
%parts

%Get the determinant of F_t_given_tm1
F_t_given_tm1_determinant = det(F_t_given_tm1);
F_t_given_tm1_determinant_log = real(log(F_t_given_tm1_determinant));

%Get the inverse of F
[F_t_given_tm1_inv,~] = matrix_inverse(F_t_given_tm1);

v = y - y_bar_t_given_tm1;
kgain = P_t_given_tm1*Psi_2_prime*F_t_given_tm1_inv;
s_bar_t_given_t = s_bar_t_given_tm1 + kgain*v;
P_t_given_t = P_t_given_tm1 - kgain*Psi_2*P_t_given_tm1;

%Value of log likelihood
%This equation is in the Dynare user guide page 81-82, Canova (Methods for
%Applied Macroeconomic Research) page 221, Kim and Nelson (State-Space
%Models with Regime Switching) page 26, and Hamilton (Time Series Analysis)
%page 385

lik = -0.5*m*log(2*pi) - 0.5*F_t_given_tm1_determinant_log - 0.5*v'*F_t_given_tm1_inv*v;

%The one-step-ahead prediction for y
y_tp1_given_t = y-v;


end

