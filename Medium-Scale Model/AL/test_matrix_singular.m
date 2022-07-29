function [test_result] = test_matrix_singular(matrix)
%This function tests a matrix for singularity and near singularity

%Input
    %matrix: a matrix
    
%Outputs:
    %test_result: equal to 1 if the matrix is singular or nearly singular, 0 otherwise
    
test_result = 0;

test_value = 1e-12;

test_1 = rcond(matrix);
test_2 = isnan(test_1);

if test_1 < test_value || test_2 == 1
    test_result = 1;
end

end

