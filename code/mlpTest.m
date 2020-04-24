function mlp_Ytest = mlpTest(weights, test)
% test feats*num data
%weights is a trained net
% y is class output

%% preallocate and set up data
test=test';
layers = length(weights)+1;
Z = cell(layers);
Z{1} = test;
%% propagate forward though each layer
for l = 2:layers
    temp = weights{l-1}'*Z{l-1};
    Z{l} = 1./(1+exp(-temp));
    %Z{l} = sigmoid(weights{l-1}'*Z{l-1});
end
%% get output from final layer and format
mlp_Ytest = Z{layers};
[~,I]=max(mlp_Ytest,[],1);
mlp_Ytest = I';
