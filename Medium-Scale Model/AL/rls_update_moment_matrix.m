function [R_t,R_t_inv,R_test] = rls_update_moment_matrix(gamma,R_tm1,x_t)
%This function updates the moment matrix used in the recursive least
%squares adaptive learning algorithm

%The recursive least squares algorithm comes from Evans and Honkapohja "Learning and Expectations
%in Macroeconomics" on pages 32-34, 334, and 349.

%Inputs:
    %gamma: adaptive learning gain
    %R_tm1: k x k initial value of the moment matrix
    %x_t: k x 1 column vector of regressors
    
%Outputs:
    %R_t: new value of the moment matrix
    %R_t_inv: inverse of R_t
    %R_test: equal to 1 if the new value of R_t has NaN or infinity elements, 0 otherwise

%Find the new moment matrix
R_t = R_tm1 + gamma*(x_t*x_t' - R_tm1);

%Set default value for R_t_inv
R_t_inv = 0;

%Tolerance used in Ridge correction mechanism.
lambda = 0.00001;

%test flag
R_test = 0;

%Test the new moment matrix for NaN and infinity elements
test_nan = test_matrix_nan(R_t);
test_inf = test_matrix_inf(R_t);

if test_nan == 1 || test_inf == 1 %The new moment matrix has NaN or infinity elements
    R_test = 1;
else
    [R_t_inv] = ridge_correction_mechanism(R_t,lambda);
end

end

