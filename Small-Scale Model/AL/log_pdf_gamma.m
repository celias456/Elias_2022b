function [log_f] = log_pdf_gamma(x,k,theta)
%Computes the log of the gamma pdf at the value of x
%Source: Greenberg, Introduction to Bayesian Econometrics, page 185 and
%Wikipedia
%k: shape parameter
%theta: scale parameter

if x > 0
    log_f = - log(gamma(k)) - k*log(theta) + (k-1)*log(x) - (x/theta);
else
    log_f = -Inf;
end

