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