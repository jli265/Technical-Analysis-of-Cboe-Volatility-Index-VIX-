
close all; clear all; clc;

% Upload the data

data=readtable('^VIX.csv');
 
% Obtain the VIX index data
prices=data{1:end,6};
r = log(  prices(2:end) ./  prices(1: end-1)  );

%Construct training period with data before April 27th.  
r_train=r(1:end,:);
T_train = length(r_train);

%----------------------------------------------------------------------
%I.	Use Historical Data to Estimate Parameters
%----------------------------------------------------------------------

parameter_names = ["mu" "delta" "phi" "beta"]';

theta_initial=[-0.0101,0.6,0.4,0]; % Initial conditions 
 
h1=var(r_train);                    % Initial value of the conditional variance 
options = optimset('Display','none','TolFun',1e-20,'TolX',1e-20); 
[theta]=fminsearch(@garchloglik,theta_initial,options,data,h1);

% ---------------------------------------------------------------------
% Let us now do inference: we need the asymptotic variance/covariance
% matrix.
%
% Below, we compute the gradient and, with the gradient, the asymptotic
% variance/covariance matrix directly (by taking derivatives "manually").
% ----------------------------------------------------------------------

h=[h1;zeros(T_train-1,1)];

sum4 = zeros(4,4);
grad=zeros(4,T_train);

mu=theta(1);
delta=theta(2);
phi=theta(3);
beta=theta(4);

 

for t=2:T_train
    
    h(t) = mu+delta*h(t-1)+phi*(r_train(t-1)-beta)^2;
    grad(1,t)=-0.5/h(t)+0.5/(h(t)^2)*(r_train(t)-beta)^2 ;
    grad(2,t)=grad(1,t)*h(t-1);
    grad(3,t)=grad(1,t)*( r_train(t-1)-beta )^2;
    grad(4,t)=(r_train(t)-beta)/h(t);
    
    sum4=sum4+grad(:,t)*grad(:,t)';
    
end
 
asy_var = (1/T_train)*inv((1/T_train)*sum4);
t_stats = theta'./sqrt(diag(asy_var));

outcome = table(parameter_names,theta',sqrt(diag(asy_var)), t_stats,...
                  'VariableNames',{'parameters' 'Estimates_MLE' 'Std Errors' 't_statistics'});
disp(outcome);

%----------------------------------------------------------------
% This is the function which obtains the log-likelihood given an assumption
% of normality on the error terms
%-----------------------------------------------------------------


%-----------------------------------------------------------------
% II.	Use Monte Carlo Simulations to Predict Future Returns
%-----------------------------------------------------------------

T_test=6;
N=1e6;
h_test=[h(end,1)*ones(1,N);zeros(T_test,N)];
epsilon=[(r_train(end,1)-beta)*ones(1,N);zeros(T_test,N)];

r_test=[r_train(end,1)*ones(1,N) ; zeros(T_test,N) ];
vix=[prices(end,1)*ones(1,N); zeros(T_test,N)];

rng(20);
u=[epsilon(1,:)./sqrt(h_test(1,:));...
   randn(T_test,N)];

for i=2:T_test+1 
    
h_test(i,:) = mu+delta.*h_test(i-1,:)+phi.*epsilon(i-1,:).^2;
epsilon(i,:)=sqrt(h_test(i,:)).*u(i,:); 
r_test(i,:)=beta+epsilon(i,:);
vix(i,:)= vix(i-1,:).*exp(r_test(i,:));

end   

vix_pred=mean(vix(end,:));
vix_pred_std=std(vix(end,:));
fprintf("Our predicted final VIX price is %.2f.\n", vix_pred);
fprintf("Our predicted interval within 5%% deviation is [%.2f, %.2f]. \n", vix_pred/1.05, vix_pred/0.95);

% Plot 6-day predictions 
figure;hold on;

plot(0:8,prices(end-8:end,1)',"ko-");
plot(8:14, real(mean(vix,2)) ,"co-");
plot(14*ones(1,2), ...
    [vix_pred/1.05, vix_pred/0.95], 'r','LineWidth',0.3);

ylabel("VIX Prices");
xlabel("Days from April 20th");
title("VIX Predictions");
legend(["actual" "predicted average" "VIX interval with 5% deviation on May 8th" ],"location","best");
grid on;
vix_mix=[ prices(end-8:end-1,1) ; real(mean(vix,2))];
for  i=0:14
    text(i,vix_mix(i+1,1) ,['' num2str(round(vix_mix(i+1,1),2))''],'FontSize',9);
    if i>8
    text(i,vix_mix(i+1,1) ,['' num2str(round(vix_mix(i+1,1),2))''],'FontSize',9,"Color","c") 
    end
end
grid off;
hold off;

%-----------------------------------------------------------------
%Embedded function
function y=garchloglik(theta,data,h1)
prices=data{1:end,6};
r = log(  prices(2:end) ./  prices(1: end-1)  );
r_train=r(1:end ,:);
T_train = length(r_train);


mu=theta(1);
delta=theta(2);
phi=theta(3);
beta=theta(4);

sum1=-T_train/2*log(2*pi);
sum2=-0.5*log(h1);
sum3=-0.5*(r_train(1)-beta)^2/h1;

h=[h1;zeros(T_train-1,1)];

for t=2:T_train
    h(t) = mu+delta*h(t-1)+phi*(r_train(t-1)-beta)^2;
    sum2=sum2-0.5*log(h(t));
    sum3=sum3-0.5*(r_train(t)-beta)^2/h(t);
end
y= -(1/T_train)*(sum1+sum2+sum3);      
end
%-----------------------------------------------------------------