function [W, M] = WeightKalmanSwitchFilter_LG(L, Z, M_prev)

%     n = size(Z,1);
%     m = zeros(size(Z));
%     LZ = L.*Z;
%     for i=1:n
%         for j=1:n
% %             m(i,j) = L(i,j)*Z(i,j)*M_prev(i);
%             m(i,j) = LZ(i,j)*M_prev(i);
%         end
%     end
    m = ((L.*Z)' * diag(M_prev))';
%     disp('vectorialized comp:')
%     norm(mm-m)
    M_ij = m./sum(sum(m));
    M = sum(M_ij,1);
% %     for i=1:size(M_ij,1)
% %         W(i,:) = M_ij(i,:)./M;
% %     end
    
    W = M_ij * diag(1./M);    
end