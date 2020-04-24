function [Yhard, Ysoft] = rfTest(model, X, opts)
% X is NxD, where rows are data points
% model comes from forestTrain()
% Yhard are hard assignments to X's, Ysoft is NxK array of
% probabilities, where there are K classes.

if nargin<3, opts= struct; end

NTrees= length(model.trees);
u= model.trees{1}.classes; % Assume at least one tree
Ysoft= zeros(size(X,1), length(u));
for i=1:NTrees
    [~, ysoft] = treeTest(model.trees{i}, X, opts);
    Ysoft= Ysoft + ysoft;
end

Ysoft = Ysoft/NTrees;
[~, ix]= max(Ysoft, [], 2);
Yhard = u(ix);
end
