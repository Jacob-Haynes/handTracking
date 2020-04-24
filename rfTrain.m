function model = rfTrain(X, Y, options)
% X is NxD, N data by D features

%X= bsxfun(@rdivide, bsxfun(@minus, X, mean(X)), var(X) + 1e-10);
% Y is labels
% best options found to be
% opts.depth= 12;
% opts.numTrees= 100; or up, more = better
% opts.numSplits= 7;

NTrees= 10; %default number of trees

if nargin < 3, options= struct; end
if isfield(options, 'NTrees'), NTrees= options.NTrees; end

trees= cell(1, NTrees);
for i=1:NTrees
    
    trees{i} = treeTrain(X, Y, options);
end

model.trees = trees;
end