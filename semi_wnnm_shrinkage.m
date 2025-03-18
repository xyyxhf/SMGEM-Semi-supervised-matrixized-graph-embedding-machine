% Haifeng Xu, Anhui University of Technology, January 2023. 
% Contact information: see readme.txt.
%
% Reference: 
% Pan H, Xu H, Zheng J, et al. A semi-supervised matrixized graph embedding machine for roller bearing 
% fault diagnosis under few-labeled samples. IEEE Transactions on Industrial Informatics.
% 
% First written by Haifeng Xu, Anhui Universiy of Technology, October 2021.

function [D, nuc_sumvalue] = semi_wnnm_shrinkage(X,tau)
C = 1;

[U, S, V] =    svd(X);
SingVals =     diag(S);
W =            CalcWeights(SingVals, C);
vSingValsB =   max(SingVals - tau .* W,0);
vNonZeroInds = (vSingValsB > 0);
D =            U(:,vNonZeroInds) * diag(vSingValsB(vNonZeroInds)) * V(:,vNonZeroInds)';
nuc_sumvalue =      sum(SingVals .* W);
end

%%
function w = CalcWeights(SingVals, C)
w = C./(SingVals + 1e-16);
end
%%