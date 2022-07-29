function [c_t] = rls_update_beliefs(gamma,c_tm1,R_t_inv,y_t,x_t)
%This function updates the belief vector used in the recursive least
%squares adaptive learning algorithm

%The recursive least squares algorithm comes from Evans and Honkapohja "Learning and Expectations
%in Macroeconomics" on pages 32-34, 334, and 349.

%Inputs:
    %gamma: adaptive learning gain
    %c_tm1: k x 1 column vector of beliefs from the previous period
    %R_t_inv: k x k matrix which is the inverse of the current period moment matrix
    %y_t: most recent observation of the variable being determined
    %x_t: k x 1 column vector of regressors
    
%Output:
    %c_t: k x 1 column vector of updated (i.e., current period) values of beliefs

%Update the belief vector
c_t = c_tm1 + gamma*R_t_inv*x_t*(y_t - x_t'*c_tm1);

end

