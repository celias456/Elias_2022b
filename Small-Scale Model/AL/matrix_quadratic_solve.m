function [PP,solution] = matrix_quadratic_solve(Psi_mat,Gamma_mat,Theta_mat)
%This function solves the matrix quadratic equation
%The solution method is from Harold Uhlig's "A Toolkit for Analyzing
%Nonlinear Dynamic Stochastic Models Easily", pages 37 through 40, and
%utilizes the code from his accompanying Matlab files.

%The matrix quadratic equation is:
% Psi_mat*PP^2 - Gamma_mat*PP - Theta_mat = 0

%Psi_mat, Gamma_mat, Theta_mat, are m x m matrices

%Output
% PP: m x m matrix PP
% solution: 1 if a solution was found, 0 if no solution was found

%Get the dimension of Psi, Gamma, and Theta
m_states = size(Gamma_mat,1);

%Initialize the P matrix
PP = zeros(m_states,m_states);

%Initialize the solution flag
solution = 1;

% Roots smaller than TOL are regarded as zero.
% Complex numbers with distance less than TOL are regarded as equal.
TOL = .000001; 

Xi_mat    = [ Gamma_mat,     Theta_mat
              eye(m_states), zeros(m_states) ];
Delta_mat = [ Psi_mat,       zeros(m_states)
              zeros(m_states), eye(m_states) ];
          
%Check the Xi_mat and Delta_mat matrices for Nan and/or Inf elements
test_Xi_mat_nan = test_matrix_nan(Xi_mat);
test_Xi_mat_inf = test_matrix_inf(Xi_mat); 
test_Delta_mat_nan = test_matrix_nan(Delta_mat);
test_Delta_mat_inf = test_matrix_inf(Delta_mat);

if test_Xi_mat_nan == 1 || test_Xi_mat_inf == 1 || test_Delta_mat_nan == 1 || test_Delta_mat_inf == 1
    
    solution = 0;

else
          
    [Xi_eigvec,Xi_eigval] = eig(Xi_mat,Delta_mat);

    if rank(Xi_eigvec) < m_states
        message = 'SOLVE.M: Sorry! Xi is not diagonalizable! Cannot solve for PP.         ';
        disp(message);

        solution = 0;

    else

        [Xi_sortabs,Xi_sortindex] = sort(abs(diag(Xi_eigval)));
        Xi_sortvec = Xi_eigvec(1:2*m_states,Xi_sortindex);
        Xi_sortval = diag(Xi_eigval(Xi_sortindex,Xi_sortindex));
        Xi_select = 1 : m_states;

        if imag(Xi_sortval(m_states)) ~= 0

            if (abs( Xi_sortval(m_states) - conj(Xi_sortval(m_states+1)) ) < TOL)
                % NOTE: THIS LAST LINE MIGHT CREATE PROBLEMS, IF THIS EIGENVALUE OCCURS MORE THAN ONCE!!
                % IF YOU HAVE THAT PROBLEM, PLEASE TRY MANUAL ROOT SELECTION.

                drop_index = 1;

                while (abs(imag(Xi_sortval(drop_index)))>TOL) && (drop_index < m_states)

                    drop_index = drop_index + 1;
                end

                if drop_index >= m_states

                    message = ['SOLVE.M: You are in trouble. You have complex eigenvalues, and I cannot'
                               '   find a real eigenvalue to drop to only have conjugate-complex pairs.'
                               '   Put differently: your PP matrix will contain complex numbers. Sorry!'];
                    disp(message); 

                    solution = 0;

                    if m_states == 1

                        message = ['   TRY INCREASING THE DIMENSION OF YOUR STATE SPACE BY ONE!            '
                                   '   WATCH SUNSPOTS!                                                     '];                     
                        disp(message);

                        solution = 0;

                    end

                else
                    message = ['SOLVE.M: I will drop the lowest real eigenvalue to get real PP.        '
                               '         I hope that is ok. You may have sunspots.                     ']; 
                    disp(message); 

                    solution = 0;

                    Xi_select = [ 1: (drop_index-1), (drop_index+1):(m_states+1)];

                end

            end

        end

        if max(Xi_select) < 2*m_states
            if Xi_sortabs(max(Xi_select)+1) < 1 - TOL

                message = ['SOLVE.M: You may be in trouble. There are stable roots NOT used for PP.'
                           '         I have used the smallest roots: I hope that is ok.            '  
                           '         If not, try manually selecting your favourite roots.          '
                           '         For manual root selection, take a look at the file solve.m    '
                           '         Watch out for sunspot solutions.                              '];
                disp(message); 

                solution = 0;

            end 

        end

        if max(abs(Xi_sortval(Xi_select)))  > 1 + TOL

            message = ['SOLVE.M: You may be in trouble.  There are unstable roots used for PP. '
                       '         Keep your fingers crossed or change your model.               '];
            disp(message);

            solution = 0;

        end

        if abs( max(abs(Xi_sortval(Xi_select))) - 1  ) < TOL

            message = ['SOLVE.M: Your matrix PP contains a unit root. You probably do not have '
                       '         a unique steady state, do you?  Should not be a problem, but  '
                       '         you do not have convergence back to steady state after a shock'
                       '         and you should better not trust long simulations.             '];
            disp(message); 

            solution = 0;

        end

        Lambda_mat = diag(Xi_sortval(Xi_select));
        Omega_mat  = (Xi_sortvec((m_states+1):(2*m_states),Xi_select));

        if rank(Omega_mat)<m_states

            message = 'SOLVE.M: Sorry! Omega is not invertible. Cannot solve for PP.          ';
            disp(message);

            solution = 0;

        else

            PP = Omega_mat*Lambda_mat/Omega_mat;
            PP_imag = imag(PP);
            PP = real(PP);

            if sum(sum(abs(PP_imag))) / sum(sum(abs(PP))) > .000001
                message = ['SOLVE.M: PP is complex.  I proceed with the real part only.            '  
                           '         Hope that is ok, but you are probably really in trouble!!     '
                           '         You should better check everything carefully and be           '
                           '         distrustful of all results which follow now.                  '];
                disp(message);

                solution = 0;

            end
        end

    end
end

end

