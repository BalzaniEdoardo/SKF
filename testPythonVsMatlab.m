clear all
close all
clc

%% Check smoothing against python implementation
load('switchTest.mat')

[x_t_t_prime, V_t_t_prime, V_tm1_t__t_prime,L_prime ] = KalmanSwitchFilter_LG(x, V, y, F, H, Q, R);
disp('FILTER STEP: ')
disp(sprintf('  delta x: %f',norm(x_t_t-x_t_t_prime)))
disp(sprintf('  delta V: %f', norm(V_t_t_prime - V_t_t)))
disp(sprintf('  delta cov: %f',norm(V_tm1_t__t-V_tm1_t__t_prime)))
disp(sprintf('  L: %f',norm(L-L_prime)))

%% Check collapse
clear all
load('collapseTest.mat')
[x_coll_prime,cov_coll_prime] = collapse_LG(X,cov,P);
disp('COLLAPSE STEP: ')
disp(sprintf('  delta x: %f',norm(x_coll-x_coll_prime)))
disp(sprintf('  delta cov: %f', norm(cov_coll_prime - cov_coll)))


%% Check weight
clear all
load('weightTest.mat')
[W_t_prime,M_t_prime] = WeightKalmanSwitchFilter_LG(L,Z,M_tm1);
disp('WEIGHT STEP: ')
disp(sprintf('  delta M: %f',norm(M_t-M_t_prime)))
disp(sprintf('  delta W: %f', norm(W_t - W_t_prime)))
