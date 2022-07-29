function [log_f] = log_pdf_inverse_gamma_1(x,s,q)
%Computes the log of the inverse gamma pdf at the value of x
%Source: YADA manual, page 60
%s: location parameter 
%q: degrees of freedom (integer) 

if x > 0
    log_f = log(2) - log(gamma(q/2)) + (q/2)*log((q*s^2)/2) - (q+1)*log(x) - ((q*s^2)/(2*x^2));
else
    log_f = -Inf;
end


end
