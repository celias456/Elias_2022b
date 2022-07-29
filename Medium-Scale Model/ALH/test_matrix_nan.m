function [test_result] = test_matrix_nan(matrix)
%This function tests a matrix to see if it contains any NaN elements

%Input
    %matrix: a matrix
    
%Output:
    %test_result: equal to 1 if matrix contains any NaN elements, 0 otherwise
    
test_result = 0;

test_1 = isnan(matrix);
test_2 = sum(test_1);
test_3 = sum(test_2);

if test_3 > 0
    test_result = 1;
end

end

