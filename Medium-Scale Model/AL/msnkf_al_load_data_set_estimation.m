function [data,Sigma_hat,c,first_observation,Sigma_u_sd,theta,prior_information,T] = msnkf_al_load_data_set_estimation(data_set_identifier)
%Small-Scale New Keynesian Model with Homogeneous Expectations

%Loads the data set and associated characteristics

%Input
%data_set_identifier: data set used for estimation
    %1 = 2001
    %2 = 2007
    %3 = 2020

%Parameters
% First entry is initial value
% Second entry is prior distribution number
% Third entry is prior distribution hyperparameter 1
% Fourth entry is prior distribution hyperparameter 2

%Output:
%data: variables in the data set
%Sigma_hat: variance of the jumping distribution used in the M-H algorithm
%c: scaling parameter used in the M-H algorithm
%first_observation: first observation used in the data set
%Sigma_u_sd: standard deviation of the measurement error term in measurement equation
%theta: vector of parameters
%prior_information: prior information for parameters in "theta" vector

if data_set_identifier == 1 %2001
    
    load('msnkf_data_2001.mat','dy','dc','dinve','dw','labobs','pinfobs','robs'); 
    
    %Covariance matrix used for the variance of the proposal distribution in the MH algorithm (obtained from Dynare)
    Sigma_hat = load('msnkf_al_sigma_hat_2001.csv'); 
    
    %Load the parameter information
    parameter_info = load('msnkf_al_parameters_2001.csv');
   
    %Scaling parameter; increase this to have a lower acceptance rate in the MH algorithm
    c = 0.0001;
    
elseif data_set_identifier == 2 %2007
    
    load('msnkf_data_2007.mat','dy','dc','dinve','dw','labobs','pinfobs','robs'); 
    
    %Covariance matrix used for the variance of the proposal distribution in the MH algorithm (obtained from Dynare)
    Sigma_hat = load('msnkf_al_sigma_hat_2007.csv'); 
    
    %Load the parameter information
    parameter_info = load('msnkf_al_parameters_2007.csv');
   
    %Scaling parameter; increase this to have a lower acceptance rate in the MH algorithm
    c = 0.000007;
    
else %2020
    
    load('msnkf_data_2020.mat','dy','dc','dinve','dw','labobs','pinfobs','robs'); 
    
    %Covariance matrix used for the variance of the proposal distribution in the MH algorithm (obtained from Dynare)
    Sigma_hat = load('msnkf_al_sigma_hat_2020.csv'); 
    
    %Load the parameter information
    parameter_info = load('msnkf_al_parameters_2020.csv');
   
    %Scaling parameter; increase this to have a lower acceptance rate in the MH algorithm
    c = 0.000003;
    
end

%Build the data set
data = [dy,dc,dinve,dw,labobs,pinfobs,robs]; 

%Number of observations in the data set
T = size(data,1);

%First observation in the data set
first_observation = 3;

%Ensures that the Sigma_hat matrix will be symmetric and postive semi-definite so that it can be used in the "mvnrnd" function 
Sigma_hat = nearestSPD(Sigma_hat);

%Standard deviation of measurement error
Sigma_u_sd = 0.0;

%Stack the parameters into a column vector
theta = parameter_info(:,1);

%Stack the prior information in a matrix
prior_information = parameter_info(:,2:4);

end