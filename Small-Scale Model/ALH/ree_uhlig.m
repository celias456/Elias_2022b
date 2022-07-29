function [Phi_1,Phi_epsilon,P,Q] = ree_uhlig(F,G,H,L,M,N,O)
%This function calculates the matrices of the rational expectations
%equilibrium solution using Uhlig's method. %The solution method is from 
%Harold Uhlig's "A Toolkit for Analyzing Nonlinear Dynamic Stochastic 
%Models Easily", pages 31 through 33.

%The log-linearized equilibrium relationships can be written as:

% 0 = E_t[F*x_{t+1} + G*x_t + H*x_{t-1} + L*z_{t+1} + M*z_{t}
% z_{t+1} = N*z_t + O*epsilon_{t+1}

%where x is a vector of endogenous state variables of size m x 1 and z is 
%a vector of exogenous stochastic processes of size k x 1.

%The solution is:
% x_t = P*x_{t-1} + Q*z_{t}
%or
% s_t = Phi_1*s_{t-1} + Phi_epsilon*epsilon_t
%
%where s = [x' z']'

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

m = size(F,1);
k = size(N,1);

I_k = eye(k);

[P,solution] = matrix_quadratic_solve(F,-G,-H);

N_transpose = transpose(N);

V = kron(N_transpose,F) + kron(I_k,F*P+G);
V_inv = matrix_inverse(V);

LNplusM = L*N + M;
[LNplusM_rows,LNplusM_columns] = size(LNplusM);
LNplusM_vec = -reshape(LNplusM,LNplusM_rows*LNplusM_columns,1);

Q_vec = V_inv*(LNplusM_vec);
Q = reshape(Q_vec,LNplusM_rows,LNplusM_columns);

%This generates the Phi_1 and Phi_epsilon matrices that match with what
%Gensys produces
Phi_1 = [P,Q*N;zeros(k,m),N];
Phi_epsilon = [Q*O;O];


end

