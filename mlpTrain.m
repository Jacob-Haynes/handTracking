function [weights,err] = mlpTrain(train, response, h, epoch, rate)
%train: feature x n train data set features
%response: class x n response classes
%h: Layer x 1 vector - number of layers with how many nodes in each

%% prep data
train=train'; %featsxN
response = response'; %format response into binary for classification
temp_res = zeros(max(response),size(response,2));
for i = 1:size(response,2)
    temp = response(i);
    temp_res(temp,i)=1;
end
response = temp_res;
h = [size(train,1);h(:);size(response,1)];%build layers input;hiddern;hideern...;output
layers = numel(h); %total number of layers
W = cell(layers-1); %weights prealocate
Z = cell(layers); %prealocate each layers output space
Z{1} = train; %initialise first layer inputs
err = zeros(1,epoch);
%% random initial weights
for j = 1:layers-1
    W{j} = randn(h(j),h(j+1)); 
end
%% for each epoch
for i = 1:epoch
    %forward pass for each layer via sigmoid
    for j = 2:layers
        temp = W{j-1}'*Z{j-1};
        Z{j} = 1./(1+exp(-temp));
        %Z{j} = sigmoid(temp);
    end
    %cost
    diff = response-Z{layers}; %calc diff between real and predicted
    err(i) = mean(dot(diff(:),diff(:))); 
    %propagate back though each layer
    for j = layers-1:-1:1
        grad = Z{j+1}.*(1-Z{j+1}); %gradient
        delta = grad.*diff;
        new_W = Z{j}*delta'; %change in weights
        W{j} = W{j}+rate*new_W; %apply learning rate and update W
        diff = W{j}*delta; %update the difference for back propagation
    end
end
%% return
err = err(1:i); 
weights = W;