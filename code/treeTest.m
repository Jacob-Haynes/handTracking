function [Yh, Ys] = treeTest(model, X, opts)
% Test a tree
    
if nargin < 3, opts= struct; end
d= model.depth;

[N, ~]= size(X);
nd= 2^d - 1;
NBranches = (nd+1)/2 - 1;
NLeafs= (nd+1)/2;

Yh= zeros(N, 1);
u= model.classes;
if nargout>1, Ys= zeros(N, length(u)); end

% if we can afford to store as non-sparse (100MB array, say), it is
% slightly faster.
%same as in train section
if storage([N nd]) < 100 
    dataix= zeros(N, nd);
else
    dataix= sparse(N, nd); 
end

% Propagate down the tree
for n = 1: NBranches
    % get relevant data at this node
    if n==1 
        rd = ones(N, 1)==1;
        Xr= X;
    else
        rd = dataix(:, n)==1;
        Xr = X(rd, :);
    end
    if size(Xr,1)==0, continue; end % empty branch situation
    
    yhat= modelTest(model.trainModels{n}, Xr, opts);
    
    dataix(rd, 2*n)= yhat;
    dataix(rd, 2*n+1)= 1 - yhat; 
end

% leafs and assign class
for n= (nd+1)/2 : nd
    ff= find(dataix(:, n)==1);
    
    hc= model.leafdist(n - (nd+1)/2 + 1, :);
    vm= max(hc);
    miopt= find(hc==vm);
    mi= miopt(randi(length(miopt), 1)); %choose a class randomly
    Yh(ff)= u(mi);
    
    if nargout > 1
        Ys(ff, :)= repmat(hc, length(ff), 1);
    end
end
