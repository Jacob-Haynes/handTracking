function yhat = modelTest(model, X, options)

if nargin < 3, options = struct; end

[N, D]= size(X);

if model.classifier== 1
    %pass down tree
    yhat = double(X(:, model.r) < model.t);
elseif model.classifier==0
    yhat=double(rand(N,1)<0.5);
else
    fprintf('Classifier = %d does not exist.\n', classifier);
end


end