function [matrix_inv,test_singular] = matrix_inverse(input_matrix)
%This function inverts a matrix. It checks for singularity and near
%singularity before the inverstion. If the matrix is singular or nearly 
%singular, it uses the pseudo-inverse function.

%Input
    %input_matrix: a square matrix
    
%Outputs
    %matrix_inv: the inverse of the input matrix
    %test_singular:  equal to 1 if the matrix is singular, 0 otherwise

test_singular = test_matrix_singular(input_matrix);

if test_singular == 0
    matrix_inv = inv(input_matrix);
else
    matrix_inv = pinv(input_matrix);
end

end

