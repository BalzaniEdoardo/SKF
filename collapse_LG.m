function [X, V] = collapse_LG(X_prev, V_prev, w_prev)
    
%     dim = size(V_prev,1)/2;
%     
%     X = [X_prev(1:dim,1)*w_prev(1,1) + X_prev(dim+1:end,1)*w_prev(2,1);...
%          X_prev(1:dim,2)*w_prev(1,2) + X_prev(dim+1:end,2)*w_prev(2,2)];
%     
%     v1 = squeeze(V_prev(1:dim,1:dim,1))*w_prev(1,1) + squeeze(V_prev(dim+1:end,dim+1:end,1))*w_prev(2,1);
%     v2 = squeeze(V_prev(1:dim,1:dim,2))*w_prev(1,2) + squeeze(V_prev(dim+1:end,dim+1:end,2))*w_prev(2,2);
%     V1 = [v1,zeros(size(v1));zeros(size(v2)),v2];
%     
%     diff1 = X_prev(1:dim,1) - X(1:dim);
%     diff2 = X_prev(dim+1:end,1) - X(1:dim);
%     diff3 = X_prev(1:dim,2) - X(dim+1:end);
%     diff4 = X_prev(dim+1:end,2) - X(dim+1:end);
%     V21 = diff1*diff1' *w_prev(1,1) + diff2*diff2' *w_prev(2,1);
%     V12 = diff3*diff3' *w_prev(1,2) + diff4*diff4' *w_prev(2,2);
%     V2 = [V21,zeros(size(V21));zeros(size(V12)),V12];
%     
%     V = V1+V2;
    
    
    X = X_prev*w_prev;
    
    V1 = squeeze(V_prev(:,:,1))*w_prev(1) + squeeze(V_prev(:,:,2))*w_prev(2);
    
    diff1 = X_prev(:,1) - X;
    diff2 = X_prev(:,2) - X;
    
    V2 = diff1*diff1' *w_prev(1) + diff2*diff2' *w_prev(2);
    
    V = V1+V2;
    
end