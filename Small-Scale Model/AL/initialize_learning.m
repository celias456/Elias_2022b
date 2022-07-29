function [s,S_1,S_2,A,B,R_A,R_B,z_A,z_B,error_count] = initialize_learning(number_endogenous_variables,number_exogenous_variables,number_state_variables,T,hee)
%Initializes the heterogeneous expectations adaptive learning algorithm for 
%a DSGE model

%Input:
%number_endogenous_variables: number of endogenous variables
%number_exogenous_variables: number of exogenous variables
%number_state_variables: number of variables in state transition equation
%T: number of time periods
%hee: structure with four elements:
    %hee.A_1_bar: HEE expressions for the coefficients on agent-type A's endogenous variable regressors
    %hee.A_2_bar: HEE expressions for the coefficients on agent-type A's exogenous variable regressors
    %hee.B_bar: HEE expressions for the coefficients on agent-type B's exogenous variable regressor
    %hee.solution: equals 1 if solution is unique and stable, 0 otherwise

%Output:
%s: matrix for endogenous variables in state-space system
%S_1: adaptive learning state-space matrix
%S_2: adaptive learning state-space matrix
%A: matrix for agent-type A beliefs
%B: matrix for agent-type B beliefs
%R_A: moment matrix for agent-type A
%R_B: moment matrix for agent-type B
%z_A: regressors for agent-type A
%z_B: regressors for agent-type B
%error_count: counts the number of iterations in which the beliefs had to be set to their previous period values
    %1st column is for NaNs or infinity in the S matrix
    %2nd column is for a non-stationary S_1 matrix

%Create storage for state variables
s = zeros(number_state_variables,T);

%Initialize the State-space matrices
S_1 = zeros(number_state_variables,number_state_variables,T);
S_2 = zeros(number_state_variables,number_exogenous_variables,T);

%Create storage for beliefs
A = zeros(number_endogenous_variables,number_endogenous_variables+number_exogenous_variables,T);
B = zeros(number_endogenous_variables,number_exogenous_variables,T);

%Initial values for beliefs (equal to HEE solution)
A(:,:,1) = [hee.A_1_bar,hee.A_2_bar];
A(:,:,2) = [hee.A_1_bar,hee.A_2_bar];
B(:,:,1) = hee.B_bar;
B(:,:,2) = hee.B_bar;

%Create storage for moment matrices
R_A = zeros(number_endogenous_variables+number_exogenous_variables,number_endogenous_variables+number_exogenous_variables,T);
R_B = zeros(number_exogenous_variables,number_exogenous_variables,T);

%Initial values for moment matrices
R_A(:,:,1) = 0.1*eye(number_endogenous_variables+number_exogenous_variables);
R_A(:,:,2) = 0.1*eye(number_endogenous_variables+number_exogenous_variables);
R_B(:,:,1) = 0.1*eye(number_exogenous_variables);
R_B(:,:,2) = 0.1*eye(number_exogenous_variables);

%Create storage for regressors
z_A = zeros(number_endogenous_variables+number_exogenous_variables,T);
z_B = zeros(number_exogenous_variables,T);

%Create storage for error count
error_count = zeros(T,2);

end

