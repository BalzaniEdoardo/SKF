close all
clear all
rng(2)
%% Dumped oscillator simulaitons
F = 0;
m=0.8;
b=0.5;
k=10;
dt = 0.001;
A1 = [1 dt;-k/m*dt -b/m*dt+1];
A2 = [1 dt;-3*k/m*dt 0.8*b/m*dt+1];
b = [0; F/m*dt];

N = 10000;
x = 1000;
xdot = 0;
xdotdot = 0;
X = zeros(2,2*N);
A = A1;
for t =1:2*N
    if t == N
        A = A2;
    end
    res = A*[x;xdot] + b;
    x = res(1,1);
    xdot = res(2,1);
    X(:,t) = [x;xdot];
end
figure
subplot(211)
hold on
tstep = dt*(1:2*N);
plot(tstep,X(1,:),'r')
%subplot(122)
title('Velocity')
plot(tstep,X(2,:),'b')
legend('Position','Velocity')


%% generate the measurement noise and measure yt
R = [5000,0;0,5000]; % measurement noise
% noise = mvnrnd(zeros(2,1),R,2*N);
C = 0.5*[1, -0.5; -0.5 +1.1];
 load('oscill_noise.mat')
 X = xpy;
% DX = X - xpy;
% figure
% plot(DX(1,:))
 yt = C*X+noise;

%yt = C*X+noise';

subplot(212)
plot(tstep,yt)
legend('Measure A','Measure B')


%% initialize Kalman filter
NState = 2;
M_prev = [0.5, 0.5];
Q = [250 ,0; 0,250];% system noise
x_prev1 = [0, 0]';
v_prev1 = eye(NState);
x_prev2 = [0.1, -0.2]';
v_prev2 = eye(NState);
Z = [0.5, 0.5; 0.5 0.5];
PostM = zeros(NState,N);
PostMeanX = zeros(NState*2,N*2);
PostCovX = zeros(NState*2,NState*2,N*2);

%% Run Switching Kalman Filter
for t=1:N*2
    t
    x_prev1_tmp = x_prev1;
    x_prev2_tmp = x_prev2;
    [x_next1, v_next1, cov_next1, Lt1] = KalmanSwitchFilter_LG(x_prev1, v_prev1, yt(:,t), A1, C, Q, R);
    [x_next2, v_next2, cov_next2, Lt2] = KalmanSwitchFilter_LG(x_prev1, v_prev1, yt(:,t), A2, C, Q, R);
    [x_next3, v_next3, cov_next3, Lt3] = KalmanSwitchFilter_LG(x_prev2, v_prev2, yt(:,t), A1, C, Q, R);
    [x_next4, v_next4, cov_next4, Lt4] = KalmanSwitchFilter_LG(x_prev2, v_prev2, yt(:,t), A2, C, Q, R);
    
    L_mat = [Lt1, Lt2; Lt3, Lt4];
    X_mat1 = [x_next1,x_next3];
    V_cube1 = cat(3,v_next1,v_next3);
    X_mat2 = [x_next2,x_next4];
    V_cube2 = cat(3,v_next2,v_next4);
    
    [W, M_prev] = WeightKalmanSwitchFilter_LG(L_mat, Z, M_prev);
    M_prev
     MM = M_prev;
    [x_1, v_1] = collapse_LG(X_mat1, V_cube1, W(:,1));
    [x_2, v_2] = collapse_LG(X_mat2, V_cube2, W(:,2));
    
    [x_prev1, v_prev1] = collapse_LG(X_mat1, V_cube1, W(:,1));
    [x_prev2, v_prev2] = collapse_LG(X_mat2, V_cube2, W(:,2));
    
    PostM(:,t) = M_prev;
    PostMeanX(1:NState,t) = x_prev1;
    PostCovX(1:NState,1:NState,t) = v_prev1;
    PostMeanX(1+NState:end,t) = x_prev2;
    PostCovX(1+NState:end,1+NState:end,t) = v_prev2;
end
    
%% plot results of filtering

figure

subplot(211)

plot(tstep,movmean(PostM(1,:),100))
hold on
title('Estimated probability')
xlabel('Time[sec]')
ylabel('Prob.')
subplot(212)

plot(tstep,X(1,:),'r')
hold on
plot(tstep,X(2,:),'b')
plot(tstep,PostMeanX(1,:))
plot(tstep,PostMeanX(2,:))
plot(tstep,PostMeanX(3,:))
plot(tstep,PostMeanX(4,:))

title('SKF tracking')
legend('Position','Velocity','Est. Position Damped','Est. Velocity Damped','Est. Velocity Amplified','Est. Velocity Amplified')
xlabel('Time[sec]')
%% zoom in 

[maxProb,argMax] = max(PostM(1,:));
idx = (argMax - 10):(argMax+10);
idx = idx+5000;
figure
%plot(tstep(idx),X(1,idx))
hold on
title('Zoom Velocity')
plot(tstep(idx),X(2,idx))
%plot(tstep(idx),PostMeanX(1,idx))
plot(tstep(idx),PostMeanX(2,idx))
%plot(tstep(idx),PostMeanX(3,idx))
plot(tstep(idx),PostMeanX(4,idx))
ylim([600,670])
xlim([5.05,5.08])
legend('Velocity','Est Vel Damp','Est Post Ampl')
xlabel('Time[sec]')

%%
%sum((PostM(1,1:N)>0.5) )/N
%sum(PostM(1,N+1:end)<0.5)/(N)