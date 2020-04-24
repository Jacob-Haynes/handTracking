function [mdl, svm_Ytest] = SvmRecognition(Xtrainfeats, Ytrain, Xtestfeats)
%use multiclass support vector machines to create a classification model 
% creates a prediction model from Xtrainfeats Ytrain
% uses model to predict Ytest from Xtestfeats and returns Ytest

mdl = fitcecoc(Xtrainfeats,Ytrain,'OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
    'expected-improvement-plus'));
svm_Ytest = predict(mdl,Xtestfeats);

end
