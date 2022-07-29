function [A_t,B_t,R_A_t,R_B_t] = ssnkf_al_update_beliefs(number_endogenous_variables,number_exogenous_variables,theta,s,A_tm1,B_tm1,R_A_tm1,R_B_tm1)
%Small-Scale New Keynesian Model with Homogeneous Expectations

%This function updates agent beliefs and moment matrices

%Input:
%number_endogenous_variables: number of endogenous variables
%number_exogenous_variables: number of exogenous variables
%theta: vector of parameters
%s: matrix of state variables
    %number of rows is the number of state variables
    %number of columns is 2
        %First column is values of state variables two periods back
        %Second column is values of state variables one period back
%A_tm1: current values of agent-type A beliefs
%B_tm1: current values of agent-type B beliefs
%R_A_tm1: current value of moment matrix for agent-type A
%R_B_tm1: current value of moment matrix for agent-type B

%Output:
%A_t: matrix of updated beliefs for agent-type A
%B_t: matrix of updated beliefs for agent-type B
%R_A_t: Updated moment matrix for agent-type A
%R_B_t: Updated moment matrix for agent-type B

%Get total number of variables
number_total_variables = number_endogenous_variables + number_exogenous_variables;

%Create storage for updated agent beliefs
A_t = zeros(number_endogenous_variables,number_total_variables);
B_t = zeros(number_endogenous_variables,number_exogenous_variables);
    
%Get value of the adaptive learning gains
gna = theta(14);
gnb = 0;

%Get values of endogenous variables one-period back 
envy_tm1 = s(1,2); 
envpi_tm1 = s(2,2);
envR_tm1 = s(3,2);

%Get new regressors
x_tm2 = s(1:number_endogenous_variables,1);
v_tm1 = s((number_endogenous_variables+1):number_total_variables,2);
z_A_tm1 = [x_tm2;v_tm1];
z_B_tm1 = v_tm1;

%Get previous values of beliefs
%Agent-type A
A_envy_tm1 = A_tm1(1,:)';
A_envpi_tm1 = A_tm1(2,:)';
A_envR_tm1 = A_tm1(3,:)';

%Agent-type B 
B_envy_tm1 = B_tm1(1,:)';
B_envpi_tm1 = B_tm1(2,:)';
B_envR_tm1 = B_tm1(3,:)';

%Update moment matrices
[R_A_t,R_A_t_inv,R_A_test] = rls_update_moment_matrix(gna,R_A_tm1,z_A_tm1);
[R_B_t,R_B_t_inv,R_B_test] = rls_update_moment_matrix(gnb,R_B_tm1,z_B_tm1);

if R_A_test == 0 && R_B_test == 0 %Moment matrices are updateable; now update the beliefs
    %1. Output
    A_envy_t = rls_update_beliefs(gna,A_envy_tm1,R_A_t_inv,envy_tm1,z_A_tm1);
    B_envy_t = rls_update_beliefs(gnb,B_envy_tm1,R_B_t_inv,envy_tm1,z_B_tm1);

    %2. Inflation
    A_envpi_t = rls_update_beliefs(gna,A_envpi_tm1,R_A_t_inv,envpi_tm1,z_A_tm1);
    B_envpi_t = rls_update_beliefs(gnb,B_envpi_tm1,R_B_t_inv,envpi_tm1,z_B_tm1);

    %3. Nominal interest rate
    A_envR_t = rls_update_beliefs(gna,A_envR_tm1,R_A_t_inv,envR_tm1,z_A_tm1);
    B_envR_t = rls_update_beliefs(gnb,B_envR_tm1,R_B_t_inv,envR_tm1,z_B_tm1);
    
    %Now store the new values of the beliefs
    % Agent-type A 
    A_t(1,:) = A_envy_t';
    A_t(2,:) = A_envpi_t';
    A_t(3,:) = A_envR_t';

    % Agent-type B
    B_t(1,:) = B_envy_t';
    B_t(2,:) = B_envpi_t';
    B_t(3,:) = B_envR_t';

else %Moment matrices are not updateable; set the beliefs and the moment matrices to the previous period values
    
    A_t = A_tm1;
    B_t = B_tm1;
    R_A_t = R_A_tm1;
    R_B_t = R_B_tm1;

end

