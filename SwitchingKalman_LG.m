clear all
close all
%% Generate data from the switching kalman filter model
% z = [differenceOfAngle, velocityOfAngle] % latent states
% y = y1 % observation
% the generative model:
% Z_ij = transition matrix of the binary S that switches between the models
% x_t = A_jt*x_(t-1) + Gauss(0,Q_j) zero mean gaussian system noise
% y_t = C_t*z_t + + Gauss(0,R) zero mean observation noise
rng(3);
dt = 0.01; % time step
NState = 2; % number of hidden state, see z above

SysNoise = 0.05;
Q = [SysNoise,0; % noise of the system
     0,SysNoise];
% QQ = eye(NState*2)*SysNoise;
 
ObsNoise = 0.05;
R = [ObsNoise]; % noise of the system
% RR = eye(NState*2)*ObsNoise;

A1 = [0,0;
      0,0]; % percieving depth of the display
A2 = [1,dt;
      0,1]; % depth is changing assuming that the velocity is constant with random noise

% AI = [A1,zeros(size(A1)); zeros(size(A2)),A2];
% AJ = [A2,zeros(size(A2)); zeros(size(A1)),A1];

C = [1,0]; % we can observe the depth but not the velocity
% CC = [C,zeros(size(C)); zeros(size(C)),C];

% The prior over latent space is Gaussian with a zero mean and broad
% variance
Z = [0.8,0.2;
     0.2,0.8];

%% generation

N=20;
S = [binornd(1,0.999,1,N), binornd(1,0.001,1,N)]+1;

step = 1;
y = zeros(1,N*2);
z = zeros(2,N*2);
z(:,1) = [0,0];

for i=1:N*2
    y(step) = mvnrnd((C*z(:,step)),R,1);
    if S(i)== 1
        z(:,step+1) = mvnrnd((A1*z(:,step))',Q,1); % Generate latent state
        step = step+1;
    elseif S(i) == 2
        z(:,step+1) = mvnrnd((A2*z(:,step))',Q,1); % Generate latent state
        step = step+1;
    end
end

figure;
hold on
plot(y(1,:),'g.')
plot(z(1,:),'b-')
plot(z(2,:),'r-')
plot(S,'ko')

%% tracking
M_prev = [0.5,0.5];
x_prev1 = [0, 0]';
v_prev1 = eye(NState);
x_prev2 = [0.1, -0.2]';
v_prev2 = eye(NState);
PostM = zeros(NState,N*2);
PostMeanX = zeros(NState*2,N*2);
PostCovX = zeros(NState*2,NState*2,N*2);

for t=1:N*2
    t
    [x_next1, v_next1, cov_next1, Lt1] = KalmanSwitchFilter_LG(x_prev1, v_prev1, y(t), A1, C, Q, R);
    [x_next2, v_next2, cov_next2, Lt2] = KalmanSwitchFilter_LG(x_prev1, v_prev1, y(t), A2, C, Q, R);
    [x_next3, v_next3, cov_next3, Lt3] = KalmanSwitchFilter_LG(x_prev2, v_prev2, y(t), A1, C, Q, R);
    [x_next4, v_next4, cov_next4, Lt4] = KalmanSwitchFilter_LG(x_prev2, v_prev2, y(t), A2, C, Q, R);
    
    L_mat = [Lt1, Lt2; Lt3, Lt4]+eps;
    X_mat1 = [x_next1,x_next3];
    V_cube1 = cat(3,v_next1,v_next3);
    X_mat2 = [x_next2,x_next4];
    V_cube2 = cat(3,v_next2,v_next4);
    
    [W, M_prev] = WeightKalmanSwitchFilter_LG(L_mat, Z, M_prev);
    
    [x_prev1, v_prev1] = collapse_LG(X_mat1, V_cube1, W(:,1));
    [x_prev2, v_prev2] = collapse_LG(X_mat2, V_cube2, W(:,2));
    
    PostM(:,t) = M_prev;
    PostMeanX(1:NState,t) = x_prev1;
    PostCovX(1:NState,1:NState,t) = v_prev1;
    PostMeanX(1+NState:end,t) = x_prev2;
    PostCovX(1+NState:end,1+NState:end,t) = v_prev2;
    
end

cols = [150/255 1/255 1/255;
    1/255 1/255 150/255;
    1/255 150/255 1/255;
    150/255 150/255 150/255];

linewidth = 2;

fig = figure('PaperPosition',[-1.2 0 19 12],'PaperOrientation','landscape', 'PaperSize',[18,12], 'PaperUnits', 'centimeters');
subplot(2,1,1)
hold on
fill([1:N*2 fliplr(1:N*2)],[(PostMeanX(1,:)-squeeze(PostCovX(1,1,:))') fliplr(PostMeanX(1,:)+squeeze(PostCovX(1,1,:))')],cols(1,:),'LineStyle','none');
fill([1:N*2 fliplr(1:N*2)],[(PostMeanX(3,:)-squeeze(PostCovX(3,3,:))') fliplr(PostMeanX(3,:)+squeeze(PostCovX(3,3,:))')],cols(2,:),'LineStyle','none');
plot(y(1,:),'o', 'Color',cols(3,:),'MarkerSize',5,'MarkerFaceColor', cols(3,:))
plot(z(1,:),'-', 'Color', cols(4,:),'LineWidth',linewidth)
plot(zeros(1,N*2),'k--','LineWidth',linewidth)
hold off
xlabel('time');
ylabel('position');
%ylim([0,1]);
%xlim([0,9]);
set(gca,'XTick',[1,10,20,30,40])
%set(gca,'XTickLabel', coh_int)
l = legend({'Staying model', 'Moving model', 'Observation', 'true state'}, 'Location','best');
set(gca,'Box','off','LineWidth',2,'FontSize',20);

subplot(2,1,2)
hold on
plot(movmean(PostM(1,:),5),'k-','LineWidth',linewidth)
plot(repmat(0.5,1,N*2),'k--','LineWidth',linewidth)
hold off
xlabel('time');
ylabel('probability of the staying model');
%ylim([0,1]);
%xlim([0,9]);
set(gca,'XTick',[1,10,20,30,40])
%set(gca,'XTickLabel', coh_int)
%legend({'Moving model','Staying model', 'Observation', 'true state'}, 'Location','NorthWest');
set(gca,'Box','off','LineWidth',2,'FontSize',20);

print(fig,['TrackingKalmanData1'],'-dpdf')