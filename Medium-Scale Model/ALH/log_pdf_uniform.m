function [log_f] = log_pdf_uniform(x,alpha,beta)
%Computes the log of the uniform pdf at the value of x
%Source: Greenberg, Introduction to Bayesian Econometrics, page 184
%alpha: lower bound
%beta: upper bound

if (alpha <= x) && (x <= beta)
    log_f = log(1/(beta-alpha));
else
    log_f = -Inf;
end

