function [test_result] = test_estability(matrix)
%This function tests a matrix for estability; it determines if all
%eigenvalues of the matrix have negative real part

%Input: 
    %matrix: a matrix
    
%Output: 
    %test_result: equal to 1 if the matrix is e-stable, 0 otherwise

test_result = 1;

test_1 = eig(matrix);
test_2 = real(test_1);
test_3 = test_2 >= 0;
test_4 = sum(test_3);

if test_4 > 0
    test_result = 0;
end

end

