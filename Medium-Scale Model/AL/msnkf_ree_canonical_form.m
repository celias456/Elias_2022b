function [Gamma_0,Gamma_1,Gamma_c,Psi,Pi] = msnkf_ree_canonical_form(number_exogenous_variables,number_jumper_variables,number_state_variables,theta)
%Medium-Scale New Keynesian Model

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
ctou = 0.025;
cg = 0.18;
clandaw = 1.5;
curvp = 10;
curvw = 10;

%Parameters
sig_ea = theta(1);
sig_eb = theta(2);
sig_eg = theta(3);
sig_eqs = theta(4);
sig_em = theta(5);
sig_epinf = theta(6);
sig_ew = theta(7);
csadjcost = theta(8);
csigma = theta(9);
chabb = theta(10);
cprobw = theta(11);
csigl = theta(12);
cprobp = theta(13);
cindw = theta(14);
cindp = theta(15);
czcap = theta(16);
cfc = theta(17);
crpi = theta(18);
crr = theta(19);
cry = theta(20);
crdy = theta(21);
%constepinf = theta(22);
constebeta = theta(23);
%constelab = theta(24);
ctrend = theta(25);
calfa = theta(26);
crhoa = theta(27);
crhob = theta(28);
crhog = theta(29);
crhoqs = theta(30);
crhoms = theta(31);
crhopinf = theta(32);
crhow = theta(33);

%Composite parameters
%cpie = 1 + (constepinf/100);
cbeta = 1/(1+constebeta/100);
cgamma = 1 + ctrend/100;
clandap = cfc;
cbetabar = cbeta*cgamma^(-csigma);
%cr = cpie/(cbeta*cgamma^(-csigma));
crk = (cbeta^(-1))*(cgamma^csigma) - (1-ctou);
cw = (calfa^calfa*(1-calfa)^(1-calfa)/(clandap*crk^calfa))^(1/(1-calfa));
cikbar = (1-(1-ctou)/cgamma);
cik = (1-(1-ctou)/cgamma)*cgamma;
clk = ((1-calfa)/calfa)*(crk/cw);
cky = cfc*(clk)^(calfa-1);
ciy = cik*cky;
ccy = 1 - cg - cik*cky;
crkky = crk*cky;
cwhlc =  (1/clandaw)*(1-calfa)/calfa*crk*cky/ccy;
%conster = (cr-1)*100;

%% Build Matrices

%Endogenous Variable Equation Numbers
eq_s_1 = 1;     %FOC labor with MPL expressed as a function of r^k and w (flexible price economy)
eq_s_2 = 2;     %FOC capacity utilization (flexible price economy)
eq_s_3 = 3;     %Firm FOC capital (flexible price economy)
eq_s_4 = 4;     %Definitiion capital services (flexible price economy)
eq_s_5 = 5;     %Investment Euler equation (flexible price economy)
eq_s_6 = 6;     %Arbitrage equation value of capital (flexible price economy)
eq_s_7 = 7;     %Consumption Euler equation (flexible price economy)
eq_s_8 = 8;     %Aggregate resource constraint (flexible price economy)
eq_s_9 = 9;     %Aggregate production function (flexible price economy)
eq_s_10 = 10;   %Wage equation (flexible price economy)
eq_s_11 = 11;   %Law of motion for capital (flexible price economy)
eq_s_12 = 12;   %FOC labor with MPL expressed as a function r^k and w 
eq_s_13 = 13;   %FOC capacity utilization
eq_s_14 = 14;   %Firm FOC capital
eq_s_15 = 15;   %Definition capital services
eq_s_16 = 16;   %Investment Euler equation
eq_s_17 = 17;   %Arbitrage equation value of capital
eq_s_18 = 18;   %Consumption Euler equation
eq_s_19 = 19;   %Aggregate resource constraint
eq_s_20 = 20;   %Aggregate production function
eq_s_21 = 21;   %New Keynesian Phillips curve
eq_s_22 = 22;   %Wage Phillips curve
eq_s_23 = 23;   %Taylor rule
eq_s_24 = 24;   %Law of motion of capital
eq_s_25 = 25;   %Law of motion for productivity shock process
eq_s_26 = 26;   %Law of motion for risk premium shock process
eq_s_27 = 27;   %Law of motion for spending shock process
eq_s_28 = 28;   %Law of motion for investment specific technology shock process
eq_s_29 = 29;   %Law of motion for monetary policy shock process
eq_s_30 = 30;   %Law of motion for price markup shock process
eq_s_31 = 31;   %Law of motion for wage markup shock process
eq_s_32 = 32;   %Auxiliary equation for lagged output (uesd for measurement equation)
eq_s_33 = 33;   %Auxiliary equation for lagged consumption (used for measurement equation)
eq_s_34 = 34;   %Auxiliary equation for lagged investment (used for measurement equation)
eq_s_35 = 35;   %Auxiliary equation for lagged real wage (used for measurement equation)
eq_s_36 = 36;   %Expectation equation for consumption
eq_s_37 = 37;   %Expectation equation for hours worked
eq_s_38 = 38;   %Expectation equation for inflation
eq_s_39 = 39;   %Expectation equation for investment
eq_s_40 = 40;   %Expectation equation for real value of existing capital stock
eq_s_41 = 41;   %Expectation equation for rental rate of capital
eq_s_42 = 42;   %Expectation equation for real wage
eq_s_43 = 43;   %Expectation equation for investment flexible price economy
eq_s_44 = 44;   %Expectation equation for real value of existing capital stock flexible price economy
eq_s_45 = 45;   %Expectation equation for rental rate of capital flexible price economy
eq_s_46 = 46;   %Expectation equation for consumption flexible price economy
eq_s_47 = 47;   %Expectation equation for hours worked flexible price economy

% Endogenous variable numbers
rrf = 1;            %Real interest rate flexible price enonomy
zcapf = 2;          %Capital utilization rate flexible price economy
rkf = 3;            %Rental rate of capital flexible price economy
kf = 4;             %Capital services flexible price economy
invef = 5;          %Investment flexible price economy
pkf = 6;            %Real value of existing capital stock flexible price economy
cf = 7;             %Consumption flexible price economy
yf = 8;             %Output flexible price economy
labf = 9;           %Hours worked flexible price economy
wf = 10;            %Real wage flexible price economy
kpf = 11;           %Capital stock flexible price economy
mc = 12;            %Gross price markup
zcap = 13;          %Capital utilization rate
rk = 14;            %Rental rate of capital
k = 15;             %Capital services
inve = 16;          %Investment
pk = 17;            %Real value of existing capital stock
c = 18;             %Consumption
y = 19;             %Output
lab = 20;           %Hours worked
pinf = 21;          %Inflation
w = 22;             %Real wage
r = 23;             %Nominal interest rate
kp = 24;            %Capital stock
a = 25;             %Productivity shock process
b = 26;             %Scaled risk premium shock process
g = 27;             %Exogenous spending shock process
qs = 28;            %Investment-specific technology shock process
ms = 29;            %Monetary policy shock process
spinf = 30;         %Price markup shock process
sw = 31;            %Wage markup shock process
y_tm1 = 32;         %Output (lagged) - auxiliary variable used for measurement equation
c_tm1 = 33;         %Consumption (lagged) - auxiliary variable used for measurement equation
inve_tm1 = 34;      %Investment (lagged) - auxiliary variable used for measurement equation
w_tm1 = 35;         %Real wage (lagged) - auxiliary variable used for measurement equation
E_c = 36;           %Consumption jumper variable
E_lab = 37;         %Hours worked jumper variable
E_pinf = 38;        %Inflation jumper variable
E_inve = 39;        %Investment jumper variable
E_pk = 40;          %Real value of existing capital stock jumper variable
E_rk = 41;          %Rental rate of capital jumper variable
E_w = 42;           %Real wage jumper variable
E_invef = 43;       %Investment flexible price economy jumper variable
E_pkf = 44;         %Real value of existing capital stock flexible price economy jumper variable
E_rkf = 45;         %Rental rate of capital flexible price economy jumper variable
E_cf = 46;          %Consumption flexible price economy jumper variable
E_labf = 47;        %Hours worked flexible price economy jumper variable

% Stochastic shock variable numbers
ea = 1;     %Productivity shock
eb = 2;     %Risk premium shock
eg = 3;     %Exogenous spending shock
eqs = 4;    %Investment specific technology shock
em = 5;     %Monetary policy shock
epinf = 6;  %Price markup shock
ew = 7;     %Wage markup shock

% Jumper variable numbers
ex_c = 1;       %Consumption
ex_lab = 2;     %Hours worked
ex_pinf = 3;    %Inflation
ex_inve = 4;    %Investment
ex_pk = 5;      %Real value of existing capital stock
ex_rk = 6;      %Rental rate of capital
ex_w = 7;       %Real wage
ex_invef = 8;   %Investment flexible price economy
ex_pkf = 9;     %Real value of existing capital stock flexible price economy
ex_rkf = 10;    %Rental rate of capital flexible price economy
ex_cf = 11;     %Consumption flexible price economy
ex_labf = 12;   %Hours worked flexible price economy


% Initialize Matrices
Gamma_0 = zeros(number_state_variables,number_state_variables);
Gamma_1 = zeros(number_state_variables,number_state_variables);
Gamma_c = zeros(number_state_variables,1);
Psi = zeros(number_state_variables,number_exogenous_variables);
Pi = zeros(number_state_variables,number_jumper_variables);

%% Endogenous Variable Equations

%Equation 1
Gamma_0(eq_s_1,rkf) = -calfa;
Gamma_0(eq_s_1,wf) = -(1-calfa);
Gamma_0(eq_s_1,a) = 1;

%Equation 2
Gamma_0(eq_s_2,zcapf) = 1;
Gamma_0(eq_s_2,rkf) = -(1/(czcap/(1-czcap)));

%Equation 3
Gamma_0(eq_s_3,rkf) = 1;
Gamma_0(eq_s_3,kf) = 1;
Gamma_0(eq_s_3,labf) = -1;
Gamma_0(eq_s_3,wf) = -1;

%Equation 4
Gamma_0(eq_s_4,zcapf) = -1;
Gamma_0(eq_s_4,kf) = 1;

Gamma_1(eq_s_4,kpf) = 1;

%Equation 5
Gamma_0(eq_s_5,invef) = 1;
Gamma_0(eq_s_5,pkf) = -(1/(1+cbetabar*cgamma))*(1/(cgamma^2*csadjcost));
Gamma_0(eq_s_5,qs) = -1;
Gamma_0(eq_s_5,E_invef) = -(1/(1+cbetabar*cgamma))*cbetabar*cgamma;

Gamma_1(eq_s_5,invef) = (1/(1+cbetabar*cgamma));

%Equation 6
Gamma_0(eq_s_6,rrf) = 1;
Gamma_0(eq_s_6,pkf) = 1;
Gamma_0(eq_s_6,b) = -(1/((1-chabb/cgamma)/(csigma*(1+chabb/cgamma))));
Gamma_0(eq_s_6,E_pkf) = -((1-ctou)/(crk+(1-ctou)));
Gamma_0(eq_s_6,E_rkf) = -(crk/(crk+(1-ctou)));

%Equation 7
Gamma_0(eq_s_7,rrf) = (1-chabb/cgamma)/(csigma*(1+chabb/cgamma));
Gamma_0(eq_s_7,cf) = 1;
Gamma_0(eq_s_7,labf) = -((csigma-1)*cwhlc/(csigma*(1+chabb/cgamma)));
Gamma_0(eq_s_7,b) = -1;
Gamma_0(eq_s_7,E_cf) = -(1/(1+chabb/cgamma));
Gamma_0(eq_s_7,E_labf) = ((csigma-1)*cwhlc/(csigma*(1+chabb/cgamma)));

Gamma_1(eq_s_7,cf) = (chabb/cgamma)/(1+chabb/cgamma);

%Equation 8
Gamma_0(eq_s_8,zcapf) = -crkky;
Gamma_0(eq_s_8,invef) = -ciy;
Gamma_0(eq_s_8,cf) = -ccy;
Gamma_0(eq_s_8,yf) = 1;
Gamma_0(eq_s_8,g) = -1;

%Equation 9
Gamma_0(eq_s_9,kf) = -cfc*calfa;
Gamma_0(eq_s_9,yf) = 1;
Gamma_0(eq_s_9,labf) = -cfc*(1-calfa);
Gamma_0(eq_s_9,a) = -cfc;

%Equation 10
Gamma_0(eq_s_10,cf) = -(1/(1-chabb/cgamma));
Gamma_0(eq_s_10,labf) = -csigl;
Gamma_0(eq_s_10,wf) = 1;

Gamma_1(eq_s_10,cf) = -(chabb/cgamma)/(1-chabb/cgamma);

%Equation 11
Gamma_0(eq_s_11,invef) = -cikbar;
Gamma_0(eq_s_11,kpf) = 1;
Gamma_0(eq_s_11,qs) = -(cikbar)*(1+cbetabar*cgamma)*(cgamma^2*csadjcost);

Gamma_1(eq_s_11,kpf) = (1-cikbar);

%Equation 12
Gamma_0(eq_s_12,mc) = 1;
Gamma_0(eq_s_12,rk) = -calfa;
Gamma_0(eq_s_12,w) = -(1-calfa);
Gamma_0(eq_s_12,a) = 1;

%Equation 13
Gamma_0(eq_s_13,zcap) = 1;
Gamma_0(eq_s_13,rk) = -(1/(czcap/(1-czcap)));

%Equation 14
Gamma_0(eq_s_14,rk) = 1;
Gamma_0(eq_s_14,k) = 1;
Gamma_0(eq_s_14,lab) = -1;
Gamma_0(eq_s_14,w) = -1;

%Equation 15
Gamma_0(eq_s_15,zcap) = -1;
Gamma_0(eq_s_15,k) = 1;

Gamma_1(eq_s_15,kp) = 1;

%Equation 16
Gamma_0(eq_s_16,inve) = 1;
Gamma_0(eq_s_16,pk) = -(1/(1+cbetabar*cgamma))*(1/(cgamma^2*csadjcost));
Gamma_0(eq_s_16,qs) = -1;
Gamma_0(eq_s_16,E_inve) = -(1/(1+cbetabar*cgamma))*(cbetabar*cgamma);

Gamma_1(eq_s_16,inve) = (1/(1+cbetabar*cgamma));

%Equation 17
Gamma_0(eq_s_17,pk) = 1;
Gamma_0(eq_s_17,r) = 1;
Gamma_0(eq_s_17,b) = -(1/((1-chabb/cgamma)/(csigma*(1+chabb/cgamma))));
Gamma_0(eq_s_17,E_pinf) = -1;
Gamma_0(eq_s_17,E_pk) = -((1-ctou)/(crk+(1-ctou)));
Gamma_0(eq_s_17,E_rk) = -(crk/(crk+(1-ctou)));

%Equation 18
Gamma_0(eq_s_18,c) = 1;
Gamma_0(eq_s_18,lab) = -((csigma-1)*cwhlc/(csigma*(1+chabb/cgamma)));
Gamma_0(eq_s_18,r) = (1-chabb/cgamma)/(csigma*(1+chabb/cgamma));
Gamma_0(eq_s_18,b) = -1;
Gamma_0(eq_s_18,E_c) = -(1/(1+chabb/cgamma));
Gamma_0(eq_s_18,E_lab) = ((csigma-1)*cwhlc/(csigma*(1+chabb/cgamma)));
Gamma_0(eq_s_18,E_pinf) = -(1-chabb/cgamma)/(csigma*(1+chabb/cgamma));

Gamma_1(eq_s_18,c) = (chabb/cgamma)/(1+chabb/cgamma);

%Equation 19
Gamma_0(eq_s_19,zcap) = -crkky;
Gamma_0(eq_s_19,inve) = -ciy;
Gamma_0(eq_s_19,c) = -ccy;
Gamma_0(eq_s_19,y) = 1;
Gamma_0(eq_s_19,g) = -1;

%Equation 20
Gamma_0(eq_s_20,k) = -cfc*calfa;
Gamma_0(eq_s_20,y) = 1;
Gamma_0(eq_s_20,lab) = -cfc*(1-calfa);
Gamma_0(eq_s_20,a) = -cfc;

%Equation 21
Gamma_0(eq_s_21,mc) = -(1/(1+cbetabar*cgamma*cindp))*((1-cprobp)*(1-cbetabar*cgamma*cprobp)/cprobp)/((cfc-1)*curvp+1);
Gamma_0(eq_s_21,pinf) = 1;
Gamma_0(eq_s_21,spinf) = -1;
Gamma_0(eq_s_21,E_pinf) = -(1/(1+cbetabar*cgamma*cindp))*(cbetabar*cgamma);

Gamma_1(eq_s_21,pinf) = (1/(1+cbetabar*cgamma*cindp))*cindp;

%Equation 22
Gamma_0(eq_s_22,c) = -(1-cprobw)*(1-cbetabar*cgamma*cprobw)/((1+cbetabar*cgamma)*cprobw)*(1/((clandaw-1)*curvw+1))*(1/(1-chabb/cgamma));
Gamma_0(eq_s_22,lab) = -(1-cprobw)*(1-cbetabar*cgamma*cprobw)/((1+cbetabar*cgamma)*cprobw)*(1/((clandaw-1)*curvw+1))*csigl;
Gamma_0(eq_s_22,pinf) = (1+cbetabar*cgamma*cindw)/(1+cbetabar*cgamma);
Gamma_0(eq_s_22,w) = 1+(1-cprobw)*(1-cbetabar*cgamma*cprobw)/((1+cbetabar*cgamma)*cprobw)*(1/((clandaw-1)*curvw+1));
Gamma_0(eq_s_22,sw) = -1;
Gamma_0(eq_s_22,E_pinf) = -(cbetabar*cgamma)/(1+cbetabar*cgamma);
Gamma_0(eq_s_22,E_w) = -(cbetabar*cgamma/(1+cbetabar*cgamma));

Gamma_1(eq_s_22,c) = -(1-cprobw)*(1-cbetabar*cgamma*cprobw)/((1+cbetabar*cgamma)*cprobw)*(1/((clandaw-1)*curvw+1))*((chabb/cgamma)/(1-chabb/cgamma));
Gamma_1(eq_s_22,pinf) = (cindw/(1+cbetabar*cgamma));
Gamma_1(eq_s_22,w) = (1/(1+cbetabar*cgamma));

%Equation 23
Gamma_0(eq_s_23,yf) = cry*(1-crr)+crdy;
Gamma_0(eq_s_23,y) = -(cry*(1-crr)+crdy);
Gamma_0(eq_s_23,pinf) = -crpi*(1-crr);
Gamma_0(eq_s_23,r) = 1;
Gamma_0(eq_s_23,ms) = -1;

Gamma_1(eq_s_23,yf) = crdy;
Gamma_1(eq_s_23,y) = -crdy;
Gamma_1(eq_s_23,r) = crr;

%Equation 24
Gamma_0(eq_s_24,inve) = -cikbar;
Gamma_0(eq_s_24,kp) = 1;
Gamma_0(eq_s_24,qs) = -cikbar*(1+cbetabar*cgamma)*(cgamma^2)*csadjcost;

Gamma_1(eq_s_24,kp) = (1-cikbar);

%Equation 25
Gamma_0(eq_s_25,a) = 1;

Gamma_1(eq_s_25,a) = crhoa;

Psi(eq_s_25,ea) = sig_ea;

%Equation 26
Gamma_0(eq_s_26,b) = 1;

Gamma_1(eq_s_26,b) = crhob;

Psi(eq_s_26,eb) = sig_eb;

%Equation 27
Gamma_0(eq_s_27,g) = 1;

Gamma_1(eq_s_27,g) = crhog;

Psi(eq_s_27,eg) = sig_eg;

%Equation 28
Gamma_0(eq_s_28,qs) = 1;

Gamma_1(eq_s_28,qs) = crhoqs;

Psi(eq_s_28,eqs) = sig_eqs; 

%Equation 29
Gamma_0(eq_s_29,ms) = 1;

Gamma_1(eq_s_29,ms) = crhoms;

Psi(eq_s_29,em) = sig_em;

%Equation 30
Gamma_0(eq_s_30,spinf) = 1;

Gamma_1(eq_s_30,spinf) = crhopinf;

Psi(eq_s_30,epinf) = sig_epinf;

%Equation 31
Gamma_0(eq_s_31,sw) = 1;

Gamma_1(eq_s_31,sw) = crhow; 

Psi(eq_s_31,ew) = sig_ew;

%Equation 32
Gamma_0(eq_s_32,y_tm1) = 1;

Gamma_1(eq_s_32,y) = 1;

%Equation 33
Gamma_0(eq_s_33,c_tm1) = 1;

Gamma_1(eq_s_33,c) = 1;

%Equation 34
Gamma_0(eq_s_34,inve_tm1) = 1;

Gamma_1(eq_s_34,inve) = 1;

%Equation 35
Gamma_0(eq_s_35,w_tm1) = 1;

Gamma_1(eq_s_35,w) = 1;

%Equation 36
Gamma_0(eq_s_36,c) = 1;

Gamma_1(eq_s_36,E_c) = 1;

Pi(eq_s_36,ex_c) = 1;

%Equation 37
Gamma_0(eq_s_37,lab) = 1;

Gamma_1(eq_s_37,E_lab) = 1;

Pi(eq_s_37,ex_lab) = 1;

%Equation 38
Gamma_0(eq_s_38,pinf) = 1;

Gamma_1(eq_s_38,E_pinf) = 1;

Pi(eq_s_38,ex_pinf) = 1;

%Equation 39
Gamma_0(eq_s_39,inve) = 1;

Gamma_1(eq_s_39,E_inve) = 1;

Pi(eq_s_39,ex_inve) = 1;

%Equation 40
Gamma_0(eq_s_40,pk) = 1;

Gamma_1(eq_s_40,E_pk) = 1;

Pi(eq_s_40,ex_pk) = 1;

%Equation 41
Gamma_0(eq_s_41,rk) = 1;

Gamma_1(eq_s_41,E_rk) = 1;

Pi(eq_s_41,ex_rk) = 1;

%Equation 42
Gamma_0(eq_s_42,w) = 1;

Gamma_1(eq_s_42,E_w) = 1;

Pi(eq_s_42,ex_w) = 1;

%Equation 43
Gamma_0(eq_s_43,invef) = 1;

Gamma_1(eq_s_43,E_invef) = 1;

Pi(eq_s_43,ex_invef) = 1;

%Equation 44
Gamma_0(eq_s_44,pkf) = 1;

Gamma_1(eq_s_44,E_pkf) = 1;

Pi(eq_s_44,ex_pkf) = 1;

%Equation 45
Gamma_0(eq_s_45,rkf) = 1;

Gamma_1(eq_s_45,E_rkf) = 1;

Pi(eq_s_45,ex_rkf) = 1;

%Equation 46
Gamma_0(eq_s_46,cf) = 1;

Gamma_1(eq_s_46,E_cf) = 1;

Pi(eq_s_46,ex_cf) = 1;

%Equation 47
Gamma_0(eq_s_47,labf) = 1;

Gamma_1(eq_s_47,E_labf) = 1;

Pi(eq_s_47,ex_labf) = 1;

end

