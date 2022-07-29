function [matrix_inv] = ridge_correction_mechanism(input_matrix,lambda)
%The function conducts a Ridge correction mechanism. If the smallest eigenvalue of the
%input matrix is less than some small value, then the matrix is modified.
%The Ridge correction mechanism is desribed in the  working paper version of 
%"Learning in an Estimated Medium-Scale DSGE Model" by  Slobodyan and Wouters (page 16, footnote 14).

%Inputs
    %input_matrix: matrix
    %lambda: tolerance for value of smallest eigenvalue of input_matrix

%Output
    %matrix_inv: inverse of input_matrix
    
test_1 = eig(input_matrix);
test_2 = min(test_1);

if test_2 < lambda
    matrix_size = size(input_matrix,1);
    I = eye(matrix_size);
    matrix_new = input_matrix+lambda*I;
    [matrix_inv,~] = matrix_inverse(matrix_new);
else
    [matrix_inv,~] = matrix_inverse(input_matrix);
end

end

