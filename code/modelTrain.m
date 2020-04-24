function model = modelTrain(X, Y, options)
% 1.look along random dimension
% maximize information gain in class labels

NSplits= 30; %default splits

if nargin < 3, options = struct; end
if isfield(options, 'classifier'), classifier = options.classifier; end
if isfield(options, 'NSplits'), NSplits = options.NSplits; end


u= unique(Y); %classes
[N, D]= size(X);

if N == 0
    % No data at leaf
    model.classifier= 0;
    return;
end
        
bestgain= -100; %info gain prep
model = struct;
% Go over classifier and generate models
for classid = classifier

    modelCandidate= struct;    
    maxgain= -1;

    if classid == 1 %was gonna try multiple methods but couldnt implement
        % optimise based on info gain 
        for q= 1:NSplits
            
            if mod(q-1,5)==0
                r= randi(D);
                col= X(:, r);
                tmin= min(col);
                tmax= max(col);
            end
            
            t= rand(1)*(tmax-tmin)+tmin;
            dec = col < t;
            IG = evaluateSplit(Y, dec, u);

            if IG>maxgain
                maxgain = IG;
                modelCandidate.r= r;
                modelCandidate.t= t;
            end
        end
    else
        fprintf('Classifier = %d does not exist.\n', classid);
    end

    % if best save for node
    if maxgain >= bestgain
        bestgain = maxgain;
        model= modelCandidate;
        model.classifier= classid;
    end

end

end

%calc Info gain
function IG= evaluateSplit(Y, dec, u)
% boolean array for left and right for class labels u

    YL= Y(dec);
    YR= Y(~dec);
    H= calcEntropy(Y, u);
    HL= calcEntropy(YL, u);
    HR= calcEntropy(YR, u);
    IG= H - length(YL)/length(Y)*HL - length(YR)/length(Y)*HR;

end

% calc entropy
function H= calcEntropy(y, u)

    classdist= histc(y, u) + 1; %histogram bins gives distribution c
    classdist= classdist/sum(classdist);
    classdist= classdist .* log(classdist);
    H= -sum(classdist); %entropy
    
end
