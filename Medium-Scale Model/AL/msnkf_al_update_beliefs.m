function [A_t,B_t,R_A_t,R_B_t] = msnkf_al_update_beliefs(number_endogenous_variables,number_exogenous_variables,theta,s,A_tm1,B_tm1,R_A_tm1,R_B_tm1)
%Medium-Scale New Keynesian Model with Homogeneous Expectations

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
gna = theta(34);
gnb = 0;

%Get values of endogenous variables one-period back 
rrf_tm1 = s(1,2); 
zcapf_tm1 = s(2,2);
rkf_tm1 = s(3,2);
kf_tm1 = s(4,2);
invef_tm1 = s(5,2);
pkf_tm1 = s(6,2);
cf_tm1 = s(7,2);
yf_tm1 = s(8,2);
labf_tm1 = s(9,2);
wf_tm1 = s(10,2);
kpf_tm1 = s(11,2);
mc_tm1 = s(12,2);
zcap_tm1 = s(13,2);
rk_tm1 = s(14,2);
k_tm1 = s(15,2);
inve_tm1 = s(16,2);
pk_tm1 = s(17,2);
c_tm1 = s(18,2);
y_tm1 = s(19,2);
lab_tm1 = s(20,2);
pinf_tm1 = s(21,2);
w_tm1 = s(22,2);
r_tm1 = s(23,2);
kp_tm1 = s(24,2);

%Get new regressors
x_tm2 = s(1:number_endogenous_variables,1);
v_tm1 = s((number_endogenous_variables+1):number_total_variables,2);
z_A_tm1 = [x_tm2;v_tm1];
z_B_tm1 = v_tm1;

%Get previous values of beliefs
%Agent-type A
A_rrf_tm1 = A_tm1(1,:)';
A_zcapf_tm1 = A_tm1(2,:)';
A_rkf_tm1 = A_tm1(3,:)';
A_kf_tm1 = A_tm1(4,:)';
A_invef_tm1 = A_tm1(5,:)';
A_pkf_tm1 = A_tm1(6,:)';
A_cf_tm1 = A_tm1(7,:)';
A_yf_tm1 = A_tm1(8,:)';
A_labf_tm1 = A_tm1(9,:)';
A_wf_tm1 = A_tm1(10,:)';
A_kpf_tm1 = A_tm1(11,:)';
A_mc_tm1 = A_tm1(12,:)';
A_zcap_tm1 = A_tm1(13,:)';
A_rk_tm1 = A_tm1(14,:)';
A_k_tm1 = A_tm1(15,:)';
A_inve_tm1 = A_tm1(16,:)';
A_pk_tm1 = A_tm1(17,:)';
A_c_tm1 = A_tm1(18,:)';
A_y_tm1 = A_tm1(19,:)';
A_lab_tm1 = A_tm1(20,:)';
A_pinf_tm1 = A_tm1(21,:)';
A_w_tm1 = A_tm1(22,:)';
A_r_tm1 = A_tm1(23,:)';
A_kp_tm1 = A_tm1(24,:)';

%Agent-type B 
B_rrf_tm1 = B_tm1(1,:)';
B_zcapf_tm1 = B_tm1(2,:)';
B_rkf_tm1 = B_tm1(3,:)';
B_kf_tm1 = B_tm1(4,:)';
B_invef_tm1 = B_tm1(5,:)';
B_pkf_tm1 = B_tm1(6,:)';
B_cf_tm1 = B_tm1(7,:)';
B_yf_tm1 = B_tm1(8,:)';
B_labf_tm1 = B_tm1(9,:)';
B_wf_tm1 = B_tm1(10,:)';
B_kpf_tm1 = B_tm1(11,:)';
B_mc_tm1 = B_tm1(12,:)';
B_zcap_tm1 = B_tm1(13,:)';
B_rk_tm1 = B_tm1(14,:)';
B_k_tm1 = B_tm1(15,:)';
B_inve_tm1 = B_tm1(16,:)';
B_pk_tm1 = B_tm1(17,:)';
B_c_tm1 = B_tm1(18,:)';
B_y_tm1 = B_tm1(19,:)';
B_lab_tm1 = B_tm1(20,:)';
B_pinf_tm1 = B_tm1(21,:)';
B_w_tm1 = B_tm1(22,:)';
B_r_tm1 = B_tm1(23,:)';
B_kp_tm1 = B_tm1(24,:)';

%Update moment matrices
[R_A_t,R_A_t_inv,R_A_test] = rls_update_moment_matrix(gna,R_A_tm1,z_A_tm1);
[R_B_t,R_B_t_inv,R_B_test] = rls_update_moment_matrix(gnb,R_B_tm1,z_B_tm1);

if R_A_test == 0 && R_B_test == 0 %Moment matrices are updateable; now update the beliefs
    %1. Real interest rate flexible price economy (rrf)
    A_rrf_t = rls_update_beliefs(gna,A_rrf_tm1,R_A_t_inv,rrf_tm1,z_A_tm1);
    B_rrf_t = rls_update_beliefs(gnb,B_rrf_tm1,R_B_t_inv,rrf_tm1,z_B_tm1);

    %2. Capital utilization rate flexible price economy (zcapf)
    A_zcapf_t = rls_update_beliefs(gna,A_zcapf_tm1,R_A_t_inv,zcapf_tm1,z_A_tm1);
    B_zcapf_t = rls_update_beliefs(gnb,B_zcapf_tm1,R_B_t_inv,zcapf_tm1,z_B_tm1);

    %3. Rental rate of capital flexible price economy (rkf)
    A_rkf_t = rls_update_beliefs(gna,A_rkf_tm1,R_A_t_inv,rkf_tm1,z_A_tm1);
    B_rkf_t = rls_update_beliefs(gnb,B_rkf_tm1,R_B_t_inv,rkf_tm1,z_B_tm1);

    %4. Capital services flexible price economy (kf)
    A_kf_t = rls_update_beliefs(gna,A_kf_tm1,R_A_t_inv,kf_tm1,z_A_tm1);
    B_kf_t = rls_update_beliefs(gnb,B_kf_tm1,R_B_t_inv,kf_tm1,z_B_tm1);

    %5. Investment flexible price economy (invef)
    A_invef_t = rls_update_beliefs(gna,A_invef_tm1,R_A_t_inv,invef_tm1,z_A_tm1);
    B_invef_t = rls_update_beliefs(gnb,B_invef_tm1,R_B_t_inv,invef_tm1,z_B_tm1);

    %6. Real value of existing capital stock flexible price economy (pkf)
    A_pkf_t = rls_update_beliefs(gna,A_pkf_tm1,R_A_t_inv,pkf_tm1,z_A_tm1);
    B_pkf_t = rls_update_beliefs(gnb,B_pkf_tm1,R_B_t_inv,pkf_tm1,z_B_tm1);

    %7. Consumption flexible price economy (cf)
    A_cf_t = rls_update_beliefs(gna,A_cf_tm1,R_A_t_inv,cf_tm1,z_A_tm1);
    B_cf_t = rls_update_beliefs(gnb,B_cf_tm1,R_B_t_inv,cf_tm1,z_B_tm1);

    %8. Output flexible price economy (yf)
    A_yf_t = rls_update_beliefs(gna,A_yf_tm1,R_A_t_inv,yf_tm1,z_A_tm1);
    B_yf_t = rls_update_beliefs(gnb,B_yf_tm1,R_B_t_inv,yf_tm1,z_B_tm1);

    %9. Hours worked flexible price economy (labf)
    A_labf_t = rls_update_beliefs(gna,A_labf_tm1,R_A_t_inv,labf_tm1,z_A_tm1);
    B_labf_t = rls_update_beliefs(gnb,B_labf_tm1,R_B_t_inv,labf_tm1,z_B_tm1);

    %10. Real wage flexible price economy (wf)
    A_wf_t = rls_update_beliefs(gna,A_wf_tm1,R_A_t_inv,wf_tm1,z_A_tm1);
    B_wf_t = rls_update_beliefs(gnb,B_wf_tm1,R_B_t_inv,wf_tm1,z_B_tm1);

    %11. Capital stock flexible price economy (kpf)
    A_kpf_t = rls_update_beliefs(gna,A_kpf_tm1,R_A_t_inv,kpf_tm1,z_A_tm1);
    B_kpf_t = rls_update_beliefs(gnb,B_kpf_tm1,R_B_t_inv,kpf_tm1,z_B_tm1);

    %12. Gross price markup (mc)
    A_mc_t = rls_update_beliefs(gna,A_mc_tm1,R_A_t_inv,mc_tm1,z_A_tm1);
    B_mc_t = rls_update_beliefs(gnb,B_mc_tm1,R_B_t_inv,mc_tm1,z_B_tm1);

    %13. Capital utilization rate (zcap)
    A_zcap_t = rls_update_beliefs(gna,A_zcap_tm1,R_A_t_inv,zcap_tm1,z_A_tm1);
    B_zcap_t = rls_update_beliefs(gnb,B_zcap_tm1,R_B_t_inv,zcap_tm1,z_B_tm1);

    %14. Rental rate of capital (rk)
    A_rk_t = rls_update_beliefs(gna,A_rk_tm1,R_A_t_inv,rk_tm1,z_A_tm1);
    B_rk_t = rls_update_beliefs(gnb,B_rk_tm1,R_B_t_inv,rk_tm1,z_B_tm1);

    %15. Capital services (k)
    A_k_t = rls_update_beliefs(gna,A_k_tm1,R_A_t_inv,k_tm1,z_A_tm1);
    B_k_t = rls_update_beliefs(gnb,B_k_tm1,R_B_t_inv,k_tm1,z_B_tm1);

    %16. Investment (inve)
    A_inve_t = rls_update_beliefs(gna,A_inve_tm1,R_A_t_inv,inve_tm1,z_A_tm1);
    B_inve_t = rls_update_beliefs(gnb,B_inve_tm1,R_B_t_inv,inve_tm1,z_B_tm1);

    %17. Real value of existing capital stock (pk)
    A_pk_t = rls_update_beliefs(gna,A_pk_tm1,R_A_t_inv,pk_tm1,z_A_tm1);
    B_pk_t = rls_update_beliefs(gnb,B_pk_tm1,R_B_t_inv,pk_tm1,z_B_tm1);

    %18. Consumption (c)
    A_c_t = rls_update_beliefs(gna,A_c_tm1,R_A_t_inv,c_tm1,z_A_tm1);
    B_c_t = rls_update_beliefs(gnb,B_c_tm1,R_B_t_inv,c_tm1,z_B_tm1);

    %19. Output (y)
    A_y_t = rls_update_beliefs(gna,A_y_tm1,R_A_t_inv,y_tm1,z_A_tm1);
    B_y_t = rls_update_beliefs(gnb,B_y_tm1,R_B_t_inv,y_tm1,z_B_tm1);

    %20. Hours worked (lab)
    A_lab_t = rls_update_beliefs(gna,A_lab_tm1,R_A_t_inv,lab_tm1,z_A_tm1);
    B_lab_t = rls_update_beliefs(gnb,B_lab_tm1,R_B_t_inv,lab_tm1,z_B_tm1);

    %21. Inflation (pinf)
    A_pinf_t = rls_update_beliefs(gna,A_pinf_tm1,R_A_t_inv,pinf_tm1,z_A_tm1);
    B_pinf_t = rls_update_beliefs(gnb,B_pinf_tm1,R_B_t_inv,pinf_tm1,z_B_tm1);

    %22. Real wage (w)
    A_w_t = rls_update_beliefs(gna,A_w_tm1,R_A_t_inv,w_tm1,z_A_tm1);
    B_w_t = rls_update_beliefs(gnb,B_w_tm1,R_B_t_inv,w_tm1,z_B_tm1);

    %23. Nominal interest rate (r)
    A_r_t = rls_update_beliefs(gna,A_r_tm1,R_A_t_inv,r_tm1,z_A_tm1);
    B_r_t = rls_update_beliefs(gnb,B_r_tm1,R_B_t_inv,r_tm1,z_B_tm1);

    %24. Capital stock (kp)
    A_kp_t = rls_update_beliefs(gna,A_kp_tm1,R_A_t_inv,kp_tm1,z_A_tm1);
    B_kp_t = rls_update_beliefs(gnb,B_kp_tm1,R_B_t_inv,kp_tm1,z_B_tm1);
    
    %Now store the new values of the beliefs
    % Agent-type A 
    A_t(1,:) = A_rrf_t';
    A_t(2,:) = A_zcapf_t';
    A_t(3,:) = A_rkf_t';
    A_t(4,:) = A_kf_t';
    A_t(5,:) = A_invef_t';
    A_t(6,:) = A_pkf_t';
    A_t(7,:) = A_cf_t';
    A_t(8,:) = A_yf_t';
    A_t(9,:) = A_labf_t';
    A_t(10,:) = A_wf_t';
    A_t(11,:) = A_kpf_t';
    A_t(12,:) = A_mc_t';
    A_t(13,:) = A_zcap_t';
    A_t(14,:) = A_rk_t';
    A_t(15,:) = A_k_t';
    A_t(16,:) = A_inve_t';
    A_t(17,:) = A_pk_t';
    A_t(18,:) = A_c_t';
    A_t(19,:) = A_y_t';
    A_t(20,:) = A_lab_t';
    A_t(21,:) = A_pinf_t';
    A_t(22,:) = A_w_t';
    A_t(23,:) = A_r_t';
    A_t(24,:) = A_kp_t';

    % Agent-type B
    B_t(1,:) = B_rrf_t';
    B_t(2,:) = B_zcapf_t';
    B_t(3,:) = B_rkf_t';
    B_t(4,:) = B_kf_t';
    B_t(5,:) = B_invef_t';
    B_t(6,:) = B_pkf_t';
    B_t(7,:) = B_cf_t';
    B_t(8,:) = B_yf_t';
    B_t(9,:) = B_labf_t';
    B_t(10,:) = B_wf_t';
    B_t(11,:) = B_kpf_t';
    B_t(12,:) = B_mc_t';
    B_t(13,:) = B_zcap_t';
    B_t(14,:) = B_rk_t';
    B_t(15,:) = B_k_t';
    B_t(16,:) = B_inve_t';
    B_t(17,:) = B_pk_t';
    B_t(18,:) = B_c_t';
    B_t(19,:) = B_y_t';
    B_t(20,:) = B_lab_t';
    B_t(21,:) = B_pinf_t';
    B_t(22,:) = B_w_t';
    B_t(23,:) = B_r_t';
    B_t(24,:) = B_kp_t';

else %Moment matrices are not updateable; set the beliefs and the moment matrices to the previous period values
    
    A_t = A_tm1;
    B_t = B_tm1;
    R_A_t = R_A_tm1;
    R_B_t = R_B_tm1;

end

