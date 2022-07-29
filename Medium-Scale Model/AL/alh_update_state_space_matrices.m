function [S_1_t,S_2_t,test,A_new,B_new,R_A_new,R_B_new] = alh_update_state_space_matrices(number_endogenous_variables,number_exogenous_variables,number_aux_variables,K_1,K_2,K_3,K_4,H_1,H_2,A_t,B_t,R_A_t,R_B_t,hee,A_tm1,B_tm1,R_A_tm1,R_B_tm1,S_1_tm1,S_2_tm1)
%This function updates the state-space matrices with new values of agent
%beliefs in a DSGE model with adaptive learning with heterogeneous
%expectations

%Input:
%number_endogenous_variables: number of endogenous variables
%number_exogenous_variables: number of exogenous variables
%number_aux_variables: number of auxiliary variables
%number_state_variables: number of state variables
%theta: vector of parameters
%K_1: matrix in condensed form equation
%K_2: matrix in condensed form equation
%K_3: matrix in condensed form equation
%K_4: matrix in condensed form equation
%H_1: matrix in condensed form equation
%A_t: agent-type A beliefs in current period
%B_t: agent-type B beliefs in current period
%R_A_t: agent-type A moment matrix in current period
%R_B_t: agent-type B moment matrix in current period
%hee: heterogeneous expectations equilibrium solution

%Output:
%S_1: matrix in state equation
%S_2: matrix in state equation
%test: flags used for NaN/inf and non-stationarity
%A_new: agent-type A updated beliefs
%B_new: agent-type B updated beliefs
%R_A_new: agent-type A updated moment matrix
%R_B_new: agent-type B updated moment matrix

%% Build the belief matrices with the current period beliefs

%Zero matrices
Zero_n = zeros(number_endogenous_variables);
Zero_m = zeros(number_exogenous_variables);
Zero_k = zeros(number_aux_variables);
Zero_mn = zeros(number_exogenous_variables,number_endogenous_variables);
Zero_nk = zeros(number_endogenous_variables,number_aux_variables);
Zero_kn = zeros(number_aux_variables,number_endogenous_variables);
Zero_mk = zeros(number_exogenous_variables,number_aux_variables);
Zero_km = zeros(number_aux_variables,number_exogenous_variables);

%Build the A_1, A_2, and B matrices with the current period beliefs
A_1 = A_t(:,1:number_endogenous_variables);
A_2 = A_t(:,number_endogenous_variables+1:end);
B = B_t;

%Build the A_1_s, A_2_s, B_1_s, and B_2_s matrices with the current period
%beliefs
A_1_s = [A_1*A_1,(A_1*A_2+A_2*H_1)*H_1,Zero_nk;Zero_mn,Zero_m,Zero_mk;Zero_kn,Zero_km,Zero_k];
A_2_s = [(A_1*A_2+A_2*H_1)*H_2;Zero_m;Zero_km];
B_1_s = [Zero_n,B*H_1*H_1,Zero_nk;Zero_mn,Zero_m,Zero_mk;Zero_kn,Zero_km,Zero_k];
B_2_s = [B*H_1*H_2;Zero_m;Zero_km];

%Build the S_1 and S_2 matrices with the current period beliefs
S_1_t = K_1*A_1_s + K_2*B_1_s + K_3;
S_2_t = K_1*A_2_s + K_2*B_2_s + K_4;

A_new = A_t;
B_new = B_t;
R_A_new = R_A_t;
R_B_new = R_B_t;

%Set the testing flags
test_naninf = 0; %testing flag for NaN or infinity elements
test_nonstationary = 0; %testing flag for nonstationarity of the solution

%% State Equation Matrices
 
%Test the S_1 and S_2 matrices to see if they have any NaN or infinity elements
[S_1_test_result_nan] = test_matrix_nan(S_1_t);
[S_1_test_result_inf] = test_matrix_inf(S_1_t);
[S_2_test_result_nan] = test_matrix_nan(S_2_t);
[S_2_test_result_inf] = test_matrix_inf(S_2_t);
 
if S_1_test_result_nan == 1 || S_1_test_result_inf == 1 || S_2_test_result_nan == 1 || S_2_test_result_inf == 1 %S_1 or S_2 has NaN and/or infinity elements; set the agent beliefs and moment matrices to the initial values and recalculate the S_1 and S_2 matrices
    
    %Set the "naninf" flag
    test_naninf = 1;

    %Set the agent beliefs and moment matrices to the initial values
    A_1 = hee.A_1_bar;
    A_2 = hee.A_2_bar;
    B = hee.B_bar;
    A_new = [hee.A_1_bar,hee.A_2_bar];
    B_new = hee.B_bar; 
    R_A_new = 0.1*eye(number_endogenous_variables+number_exogenous_variables);
    R_B_new = 0.1*eye(number_exogenous_variables);

    %Build the A_1_s, A_2_s, B_1_s and B_2_s matrices
    A_1_s = [A_1*A_1,(A_1*A_2+A_2*H_1)*H_1,Zero_nk;Zero_mn,Zero_m,Zero_mk;Zero_kn,Zero_km,Zero_k];
    A_2_s = [(A_1*A_2+A_2*H_1)*H_2;Zero_m;Zero_km];
    B_1_s = [Zero_n,B*H_1*H_1,Zero_nk;Zero_mn,Zero_m,Zero_mk;Zero_kn,Zero_km,Zero_k];
    B_2_s = [B*H_1*H_2;Zero_m;Zero_km];
 
    %Build the S_1 and S_2 matrices
    S_1_t = K_1*A_1_s + K_2*B_1_s + K_3;
    S_2_t = K_1*A_2_s + K_2*B_2_s + K_4;
   
else %S does not have any NaN or infinity elements
    
    % Now test the S_1 matrix for stationarity 
    [test_result_nonstationarity] = test_matrix_nonstationarity(S_1_t);
    
    if test_result_nonstationarity == 1 %S_1 matrix is not stationary; set the agent beliefs, moment matrices, and S_1 and S_2 matrices to the previous period values
        
        %Set the "naninf" flag
        test_nonstationary = 1;
        
        %Set the agent beliefs, moment matrices, and S_1 and S_2 matrices to the previous period values
        A_new = A_tm1;
        B_new = B_tm1;
        R_A_new = R_A_tm1;
        R_B_new = R_B_tm1;
        S_1_t = S_1_tm1;
        S_2_t = S_2_tm1;
     
    end
    
end  

%Record the test results
test = [test_naninf,test_nonstationary];

end
