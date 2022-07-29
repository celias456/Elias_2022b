function [log_prior_value] = log_prior_calculate(parameters,prior_information)
%Computes the log prior of a vector of parameters
%parameters: n*1 vector of parameters
%prior_information: n*3 matrix of prior information
    %Column 1: prior type
        %1: Beta distribution (alpha,beta)
        %2: Gamma distribution (k,theta)
        %3: Normal distribution (mu,sigma)
        %4: Inverse Gamma-1 distribution (s,q (also referred to as nu))
        %5: Uniform distribution (alpha,beta)
    %Column 2: hyperparameter 1
    %Column 3: hyperparameter 2

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%Get number of parameters
n = size(parameters,1);

%Create storage for log prior
log_prior_individual_values = zeros(n,1);

for index_1 = 1:n
    %Beta prior
    if prior_information(index_1,1) == 1         
        %Calculate the value of the log prior
        log_prior_individual_values(index_1) = log_pdf_beta(parameters(index_1),prior_information(index_1,2),prior_information(index_1,3)); 
        
    %Gamma prior
    elseif prior_information(index_1,1) == 2 
        %Calculate the value of the log prior
        log_prior_individual_values(index_1) = log_pdf_gamma(parameters(index_1),prior_information(index_1,2),prior_information(index_1,3)); 
    
    %Normal prior
    elseif prior_information(index_1,1) == 3 %Normal prior
        
        %Calculate the value of the log prior
        log_prior_individual_values(index_1) = log_pdf_normal(parameters(index_1),prior_information(index_1,2),prior_information(index_1,3));

    elseif prior_information(index_1,1) == 4 %Inverse Gamma-1 prior
        
        %Calculate the value of the log prior
        log_prior_individual_values(index_1) = log_pdf_inverse_gamma_1(parameters(index_1),prior_information(index_1,2),prior_information(index_1,3));

    elseif prior_information(index_1,1) == 5 %Uniform prior
        
        %Calculate the value of the log prior
        log_prior_individual_values(index_1) = log_pdf_uniform(parameters(index_1),prior_information(index_1,2),prior_information(index_1,3));

    end
        
end

log_prior_value = sum(log_prior_individual_values);