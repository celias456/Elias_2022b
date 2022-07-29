function [log_f] = log_pdf_normal(x,mu,sigma)
%Computes the log of the normal pdf at the value of x
%Source: Greenberg, Introduction to Bayesian Econometrics, page 187
%mu: mean
%sigma_squared: variance

log_f =  - (0.5)*(log(2) + log(pi) + 2*log(sigma)) - ((x-mu)^2)/(2*(sigma^2));


end

