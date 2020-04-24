function model = treeTrain(X, Y, options)
% Train a  tree

d= 5; % max depth default
if nargin < 3, options= struct; end
if isfield(options, 'depth'), d= options.depth; end

u= unique(Y); %number of classes
[N, ~]= size(X);
nd= 2^d - 1;
Nbranches = (nd+1)/2 - 1;
NLeafs= (nd+1)/2;

trainModels= cell(1, Nbranches); 
% if memory is small enough store as non sparse for speed 
% this little snippet of code i found online in other RF implementations
if storage([N nd]) < 100 
    dataix= zeros(N, nd); %indexs
else
    dataix= sparse(N, nd); 
end
    
leafdist= zeros(NLeafs, length(u)); % leaf distribution
% Propagate down tree while training at each node
for n = 1: Nbranches
    % get data at node
    if n==1 
        rd = ones(N, 1)==1; %relavent data
        Xr= X;
        Yr= Y;
    else
        rd = dataix(:, n)==1;
        Xr = X(rd, :);
        Yr = Y(rd);
    end
    
    % train model
    trainModels{n}= modelTrain(Xr, Yr, options);
    
    % split data to child nodes
    yhat= modelTest(trainModels{n}, Xr, options);
    
    dataix(rd, 2*n)= yhat;
    dataix(rd, 2*n+1)= 1 - yhat; %yhat{0,1} double
end

% for leaf nodes assign class
for n= (nd+1)/2 : nd %node
    rd= dataix(:, n); %relevant data
    hc = histc(Y(rd==1), u); %histagram bins
    hc = hc + 1; % prior
    leafdist(n - (nd+1)/2 + 1, :)= hc / sum(hc); %leaf distribution with bins
end

model.leafdist= leafdist;
model.depth= d;
model.classes= u;
model.trainModels= trainModels;

end
