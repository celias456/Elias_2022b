function [log_f] = log_pdf_beta(x,alpha,beta)
%Computes the log of the beta pdf at the value of x
%Source: Greenberg, Introduction to Bayesian Econometrics, page 186
%alpha: shape parameter
%beta: rate parameter

if (x > 0) && (x < 1)
    log_f = log(gamma(alpha + beta)) - log(gamma(alpha)) - log(gamma(beta)) + (alpha-1)*log(x) + (beta-1)*log(1-x); 
else
    log_f = -Inf;
end

