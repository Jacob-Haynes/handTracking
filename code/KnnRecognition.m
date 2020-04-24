function [mdl, knn_Ytest] = KnnRecognition(Xtrainfeats,Ytrain,Xtestfeats)
%use k nearest neighbor to create a classification model 
% creates a prediction model from Xtrain Ytrain
% uses model to predict Ytest from Xtest and returns Ytest

mdl = fitcknn(Xtrainfeats,Ytrain, 'OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions', ...
    struct('AcquisitionFunctionName','expected-improvement-plus'));
knn_Ytest = predict(mdl,Xtestfeats);

end
