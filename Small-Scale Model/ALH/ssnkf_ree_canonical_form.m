function [Gamma_0,Gamma_1,Gamma_c,Psi,Pi] = ssnkf_ree_canonical_form(number_exogenous_variables,number_jumper_variables,number_state_variables,theta)
%Small-Scale New Keynesian Model

%This function puts the small-scale New Keynesian model into Sims' canonical form

%Input:
%number_endogenous_variables: Number of endogenous variables
%number_exogenous_variables: Number of exogenous variables
%number_aux_variables: Number of auxiliary variables
%number_jumper_variables: Number of jumper variables
%theta: Column vector of parameters

%Canonical form
%Gamma_0*s_t = Gamma_1*s_{t-1} + Gamma_c + Psi*epsilon_t + Pi*eta_t


%% Parameters

%Fixed Parameters
parrhomp = 0;

%Parameters
parsigmag = theta(1);
parsigmaz = theta(2);
parsigmaR = theta(3);
partau = theta(4);
parkappa = theta(5);
parrhoR = theta(6);
parpsi1 = theta(7);
parpsi2 = theta(8);
parrhog = theta(9);
parrhoz = theta(10);
parrA = theta(11);

%Composite parameters
parbeta = 1/(1+(parrA/400));

%% Build Matrices

%Endogenous Variable Equation Numbers
eq_s_1 = 1;     %IS curve
eq_s_2 = 2;     %New Keynesian Phillips curve
eq_s_3 = 3;     %Monetary policy rule
eq_s_4 = 4;     %Law of motion for exogenous spending shock process
eq_s_5 = 5;     %Law of motion for technology shock process
eq_s_6 = 6;     %Law of motion for monetary policy shock process
eq_s_7 = 7;     %Auxiliary equation for lagged output for measurement equation
eq_s_8 = 8;     %Expectational equation for output
eq_s_9 = 9;     %Expectational equation for inflation 

% Endogenous variable numbers
envy = 1;           %Output
envpi = 2;          %Inflation
envR = 3;           %Nominal interest rate
exvg = 4;           %Exogenous spending shock process
exvz = 5;           %Technology shock process
exvmp = 6;          %Monetary policy shock process
auxvy = 7;          %Output (lagged) - auxiliary variable for measurement equation
epy = 8;            %Output jumper variable
eppi = 9;           %Inlation jumper variable

% Stochastic shock variable numbers
shg = 1;
shz = 2;
shR = 3;

% Jumper variable numbers
eey = 1;
eepi = 2;

% Initialize Matrices
Gamma_0 = zeros(number_state_variables,number_state_variables);
Gamma_1 = zeros(number_state_variables,number_state_variables);
Gamma_c = zeros(number_state_variables,1);
Psi = zeros(number_state_variables,number_exogenous_variables);
Pi = zeros(number_state_variables,number_jumper_variables);

%% Endogenous Variable Equations

%Equation 1
Gamma_0(eq_s_1,envy) = 1;
Gamma_0(eq_s_1,envR) = (1/partau);
Gamma_0(eq_s_1,exvg) = -(1-parrhog);
Gamma_0(eq_s_1,exvz) = -(1/partau)*parrhoz;
Gamma_0(eq_s_1,epy) = -1;
Gamma_0(eq_s_1,eppi) = -(1/partau);

%Equation 2
Gamma_0(eq_s_2,envy) = -parkappa;
Gamma_0(eq_s_2,envpi) = 1;
Gamma_0(eq_s_2,exvg) = parkappa;
Gamma_0(eq_s_2,eppi) = -parbeta;

%Equation 3
Gamma_0(eq_s_3,envy) = -(1-parrhoR)*parpsi2;
Gamma_0(eq_s_3,envpi) = -(1-parrhoR)*parpsi1;
Gamma_0(eq_s_3,envR) = 1;
Gamma_0(eq_s_3,exvg) = (1-parrhoR)*parpsi2;
Gamma_0(eq_s_3,exvmp) = -1;

Gamma_1(eq_s_3,envR) = parrhoR;

%Equation 4
Gamma_0(eq_s_4,exvg) = 1;

Gamma_1(eq_s_4,exvg) = parrhog;

Psi(eq_s_4,shg) = parsigmag;

%Equation 5
Gamma_0(eq_s_5,exvz) = 1;

Gamma_1(eq_s_5,exvz) = parrhoz;

Psi(eq_s_5,shz) = parsigmaz;

%Equation 6
Gamma_0(eq_s_6,exvmp) = 1;

Gamma_1(eq_s_6,exvmp) = parrhomp;

Psi(eq_s_6,shR) = parsigmaR;

%Equation 7
Gamma_0(eq_s_7,auxvy) = 1;

Gamma_1(eq_s_7,envy) = 1;

%Equation 8
Gamma_0(eq_s_8,envy) = 1;

Gamma_1(eq_s_8,epy) = 1;

Pi(eq_s_8,eey) = 1;

%Equation 9
Gamma_0(eq_s_9,envpi) = 1;

Gamma_1(eq_s_9,eppi) = 1;

Pi(eq_s_9,eepi) = 1;

end

