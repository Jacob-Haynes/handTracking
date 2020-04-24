clear all 
close all 
clc


%%%Uncomment bellow to visualize a sequence 
%visualize('P3_2_8_p29')

bool_exist=exist('X_train.mat');
if bool_exist
    load('X_train.mat')
else
    importData('trainingGestureData.mat','trainingData');
    X_train=prepare_data(trainingData);
    save('X_train.mat','X_train')
end


bool_exist=exist('X_test.mat');
if bool_exist
    load('X_test.mat')
else
importData('testGestureData.mat','testData');
X_test=prepare_data(testData);
save('X_test.mat','X_test')
end

bool_exist=exist('X_valid.mat');
if bool_exist
    load('X_valid.mat')
else
    importData('validationGestureData.mat','validationData');
    X_valid=prepare_data(validationData);
    save('X_valid.mat','X_valid')
end

%%X_train is 1*12cell, the 12 cells represents the 12 gestures, inside each
%%of those cells  there is n sequences of an individual doing the
%%corresponding gesture. Each one of those n sequences is a matrix of size
%%60*m, m is the number of frames, 60=20*3 are the (x,y,z) positions of the
%%20 joint locations 



%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Feature Extraction%%% 
%%%%%%%%%%%%%%%%%%%%%%%%%


%%Feature extraction used in '''Human Action Recognition Using a Temporal Hierarchy
%%of Covariance Descriptors on 3D Joint Locations'''.
%%In this implementation just one level of C is taken into account


if exist('FinalData.mat')
    load('FinalData.mat')
else
    display('Extracting Cov3dj features for X_train')
    [X_train_feats,Y_train]=extractCov3dJ(X_train);
    display('Extracting Cov3dj features for X_valid')
    [X_valid_feats,Y_valid]=extractCov3dJ(X_valid);
    display('Extracting Cov3dj features for X_test')
    [X_test_feats,Y_test]=extractCov3dJ(X_test);
    save('FinalData.mat','X_train_feats','X_valid_feats','X_test_feats','Y_train','Y_valid','Y_test')
end





%%%ACTION RECOGNITION MODELS BELLOW 
%% Knn
[knn_mld, knn_Ytest] = KnnRecognition(X_train_feats, Y_train, X_test_feats);
knn_err = 100*sum(knn_Ytest == Y_test)./numel(Y_test);
fprintf('K-Nearest Neighbor with K = %i, distance measure = %s, has %f percent correct \n',...
    knn_mld.NumNeighbors, knn_mld.Distance, knn_err)
%% SVM
[svm_mdl, svm_Ytest] = SvmRecognition(X_train_feats, Y_train, X_test_feats);
svm_err = 100*sum(svm_Ytest == Y_test)./numel(Y_test);
fprintf('Support Vector Machine multiclass (ecoc) has %f percent correct\n', svm_err)
% MLP
h = [48,24,12];
epoch = 8000;
rate = 0.0002;
[mlp_weights, mlp_train_err] = mlpTrain(X_train_feats,Y_train,h, epoch, rate);
mlp_Ytest = mlpTest(mlp_weights,X_test_feats);
mlp_err = 100*sum(mlp_Ytest == Y_test)./numel(Y_test);
fprintf('MLP has %f percent correct\n', mlp_err)
        
%% RF
options = struct;
options.depth=12;
options.NTrees = 300;
options.NSplits=5;
options.classifier=1; %for learning model but only one was implemented in the end
m = rfTrain(X_train_feats, Y_train, options);
RF_Ytest = rfTest(m,X_test_feats);
RF_err = 100*sum(RF_Ytest == Y_test)./numel(Y_test);
fprintf('Random Forest has %f percent correct\n', RF_err)
%% CNN
% form sequence data
X_train_cnn = {};
X_test_cnn = {};
for i = 1:12 %number of gestures/cells in X_train
    X_train_cnn = [X_train_cnn ; X_train{i}];
    X_test_cnn = [X_test_cnn ; X_test{i}];
end

Y_train_cnn = categorical(Y_train);
Y_test_cnn = categorical(Y_test);
inputSize = size(X_train_cnn{1}, 1);
numHiddenUnits = 100;
numClasses = max(Y_train);
cnn_layers = [sequenceInputLayer(inputSize), ...
    lstmLayer(numHiddenUnits,'OutputMode','last')...
    fullyConnectedLayer(numClasses)...
    softmaxLayer...
    classificationLayer];
maxEpochs = 100;
miniBatchSize=100;
cnn_opt = trainingOptions('sgdm',...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize);
cnnNet=trainNetwork(X_train_cnn,Y_train_cnn,cnn_layers,cnn_opt);
miniBatchSize=30;
cnn_Ytest = classify(cnnNet, X_test_cnn,'MiniBatchSize',miniBatchSize);
cnn_err = 100*sum(cnn_Ytest == Y_test_cnn)./numel(Y_test_cnn);
fprintf('CNN has %f percent correct\n', cnn_err)