load('ex4data1.mat');
x=zeros(5000,20,20);
for i=1:5000
x(i,:,:)=reshape(X(i,:),20,20);
end

digitDatasetPath = fullfile(matlabroot,'toolbox','nnet', ...
    'nndemos','nndatasets','DigitDataset');
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

options = trainingOptions('sgdm', ...
    'MaxEpochs',40,...
    'InitialLearnRate',1e-4, ...
    'Verbose',false, ...
    'Plots','training-progress');

numTrainingFiles = 1000;
[imdsTrain,imdsTest] = splitEachLabel(imds,numTrainingFiles,'randomize');

net = trainNetwork(imdsTrain,lgraph_8,options);

%trainedNet = trainNetwork(x,y,lgraph_4,options);

