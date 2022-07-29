function [mh_theta,mh_theta_A,mh_theta_B,mh_theta_R_A,mh_theta_R_B,mh_theta_s,mh_theta_P,data_forecast_sample,T,H,TpH,Sigma_u_sd] = ssnkf_al_load_data_set_forecast(data_set_identifier)
%Small-Scale New Keynesian Model with Heterogeneous Expectations

%Loads the data set and associated characteristics

%Input
%data_set_identifier: data set used for estimation
    %1 = 2001
    %2 = 2007
    %3 = 2020
    
%Output:
%parameter_info: draws of parameters from the Metropolis-Hastings algorithm
%data_estimation_sample: the estimation sample
%data_forecast_sample: the forecast sample
%TpH: total number of observations in the estimation and forecast sample
%combined
%Sigma_u_sd: standard deviation of measurement error

if data_set_identifier == 1 %2001
    
    %Load the MH draws
    load('ssnkf_al_mh_draws_2001.mat','mh_theta','mh_theta_A','mh_theta_B','mh_theta_R_A','mh_theta_R_B','mh_theta_s','mh_theta_P','T');
           
elseif data_set_identifier == 2 %2007
    
    %Load the MH draws
    load('ssnkf_al_mh_draws_2007.mat','mh_theta','mh_theta_A','mh_theta_B','mh_theta_R_A','mh_theta_R_B','mh_theta_s','mh_theta_P','T'); 
    
else %2020
    
    %Load the MH draws
    load('ssnkf_al_mh_draws_2020.mat','mh_theta','mh_theta_A','mh_theta_B','mh_theta_R_A','mh_theta_R_B','mh_theta_s','mh_theta_P','T');
    
end

%Number of periods-ahead to forecast
H = 6; 
    
%Load the data set
load('ssnkf_data.mat','obsgry','obspi','obsR');

%Get the forecast sample
data_forecast_sample = [obsgry(T+1:T+H),obspi(T+1:T+H),obsR(T+1:T+H)];

%Total number of observations in full data set
%TpH = T + H;
TpH = 2 + H;

%Standard deviation of measurement error
Sigma_u_sd = 0.0;

end

