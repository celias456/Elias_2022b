function [test_result] = test_matrix_nonstationarity(matrix)
%This function tests a matrix for nonstationarity; it checks to see if the
%matrix contains any eigenvalues that are greater than 1 in absolute value.

%Input
    %matrix: a matrix
    
%Output:
    %test_result: equal to 1 if the real parts of all of the matrix's
    %eigenvalues are greater than 1 (i.e., matrix is nonstationary), 0 otherwise
    
test_result = 0;

test_1 = eig(matrix);
test_2 = real(test_1);
test_3 = abs(test_2);
test_4 = test_3 >= 1;
test_5 = sum(test_4);

if test_5 > 0 %matrix is non-stationary
    test_result = 1;
end

end

