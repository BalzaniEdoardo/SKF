function [x_next, v_next, cov_next, Lt] = KalmanSwitchFilter_LG(x_prev, v_prev, yt, At, Ct, Qt, Rt)    
    
%     dim = size(x_prev,1)/2;
%     %prediction step
%     x_predict = At*x_prev;
%     v_predict = At*v_prev*At' + Qt;
%     %computing error
%     err = yt - Ct*x_predict;
%     %err(2) = 0;err(4) = 0;
%     %computing kalman gain
%     St = Ct*v_predict*Ct' +Rt;
%     Kt = v_predict*Ct'*pinv(St);
%     %compute L
%     Lt = [mvnpdf(err(1:dim)', zeros(1,dim), St(1:dim,1:dim));...
%           mvnpdf(err(dim+1:end)', zeros(1,dim), St(dim+1:end,dim+1:end))];
%     %update step
%     x_next = x_predict + Kt*err;
%     v_next = v_predict - Kt*St*Kt';
%     cov_next = (eye(size(Kt,1)) - Kt*Ct)*At*v_prev;
    
    %prediction step
    x_predict = At*x_prev;
    v_predict = At*v_prev*At' + Qt;
    %computing error
    err = yt - Ct*x_predict;
    %computing kalman gain
    St = Ct*v_predict*Ct' +Rt;
    Kt = v_predict*Ct'*pinv(St);
    %compute L
    %Lt = normpdf(err(1), 0, St(1,1));
    Lt = mvnpdf(err', zeros(1,length(err)), St);
    %update step
    x_next = x_predict + Kt*err;
    v_next = v_predict - Kt*St*Kt';
    cov_next = (eye(size(Kt,1)) - Kt*Ct)*At*v_prev;

end