function [hee] = hee_solution(n,m,G_1,G_2,G_3,G_4,H_1,Sigma_v)
%This function calculates the heterogeneous expectations equilibrium
%solution following the method in Berardi (2009).

%The model is in the form
%x_t = G_1*E_t^A x_{t+1} + G_2*E_t^B x_{t+1} + G_3*x_{t-1} + G_4 v_t
%v_t = H_1*v_{t-1} + H_2*epsilon_t

%x is an nx1 vector of endogenous variables, v is an mx1 vector of exogenous
%variables, and epsilon_t is i.i.d.

%Sigma_v is the asymptotic covariance matrix of v

%% Initialize variables

hee.A_1_bar = 0;
hee.A_2_bar = 0;
hee.B_bar = 0;
hee.solution = 0;

%% Create Identity Matrices

I_n = eye(n);
I_m = eye(m);
I_nm = eye(n*m);

%% Solve for A_bar using the matrix quadratic solution function

[A_1_bar,solution] = matrix_quadratic_solve(G_1,I_n,-G_3);

%% Only continue if a solution to the matrix quadratic equation was found

if solution == 1 %A solution to the matrix quadratic equation was found
    
    %Solve for A_2 and B    
    [Sigma_v_inv,~] = matrix_inverse(Sigma_v);

    %Calculate the P matrix
    P_1_hee = kron(G_3+G_1*A_1_bar*A_1_bar,Sigma_v_inv);
    P_2_hee = (I_nm - kron(G_3+G_1*A_1_bar*A_1_bar,H_1));
    [P_2_inv_hee,~] = matrix_inverse(P_2_hee);
    P_3_hee = kron(I_n,H_1*Sigma_v);
    P = P_1_hee*P_2_inv_hee*P_3_hee;

    %Calculate the Gamma matrix
    Gamma_1_hee = kron(G_1*A_1_bar,I_m) + kron(G_1,H_1') - I_nm;
    Gamma_2_hee = kron(G_2,H_1');
    Gamma_3_hee = (I_nm + P)*(kron(G_1*A_1_bar,I_m) + kron(G_1,H_1')); 
    Gamma_4_hee = (I_nm + P)*(kron(G_2,H_1')) - I_nm;
    Gamma_hee = [Gamma_1_hee,Gamma_2_hee;Gamma_3_hee,Gamma_4_hee];

    %Test for uniqueness of solution (i.e., invertibility of the Gamma matrix)
    test_unique = test_matrix_singular(Gamma_hee);
    
    if test_unique == 0 %Solution is unique
        
        %Get A_2_bar and B_bar
        Psi_1_hee = reshape(G_4',n*m,1);
        Psi_2_hee = (I_nm + P)*reshape(G_4',n*m,1);
        Psi_hee = [Psi_1_hee;Psi_2_hee];
        solution_1 = -Gamma_hee\Psi_hee;
        solution_2 = solution_1(1:n*m);
        A_2_bar = reshape(solution_2,m,n)';
        solution_3 = solution_1((n*m)+1:end);
        B_bar = reshape(solution_3,m,n)';

        %Test for E-stability of the solution
        jac = kron(A_1_bar',G_1) + kron(I_n,G_1*A_1_bar-I_n);
        jac_estab = test_estability(jac);
        Gamma_hee_estab = test_estability(Gamma_hee);
        
        if jac_estab == 1 && Gamma_hee_estab == 1 %Solution is e-stable; now record the HEE solution
            
            %Record HEE solution
            hee.A_1_bar = A_1_bar;
            hee.A_2_bar = A_2_bar;
            hee.B_bar = B_bar;
            hee.solution = 1;
            
        end
        
    end
    
end

end
