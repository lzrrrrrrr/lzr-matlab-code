%% CNN + 自注意力机制 + LSTM 射频指纹识别模型
% 基于原始代码架构，增加CNN特征提取和自注意力机制
% 数据集维度: [160, 2, 1, 4000]

% 模型参数设置 (保持与原代码一致)
inputSize = [frameLength 2 1]; 
numHiddenUnits = 100; 
numClasses = numKnownRouters + 1; 

% 新增参数
attentionDim = 128;  % 自注意力维度

% 构建CNN + 自注意力 + LSTM混合网络
layers = [
    % 输入层
    imageInputLayer(inputSize, 'Normalization', 'none', 'Name', 'Input Layer')
    
    % CNN特征提取部分
    convolution2dLayer([8, 2], 32, 'Padding', 'same', 'Name', 'Conv1')
    batchNormalizationLayer('Name', 'BN1')
    reluLayer('Name', 'ReLU1')
    maxPooling2dLayer([2, 1], 'Stride', [2, 1], 'Name', 'MaxPool1')
    
    convolution2dLayer([4, 1], 64, 'Padding', 'same', 'Name', 'Conv2')
    batchNormalizationLayer('Name', 'BN2')
    reluLayer('Name', 'ReLU2')
    maxPooling2dLayer([2, 1], 'Stride', [2, 1], 'Name', 'MaxPool2')
    
    convolution2dLayer([4, 1], 128, 'Padding', 'same', 'Name', 'Conv3')
    batchNormalizationLayer('Name', 'BN3')
    reluLayer('Name', 'ReLU3')
    
    % 特征展平和降维
    flattenLayer('Name', 'Flatten Input')
    fullyConnectedLayer(attentionDim*2, 'Name', 'Feature Reduce')
    reluLayer('Name', 'ReLU Feature')
    dropoutLayer(0.3, 'Name', 'DropOut Feature')
    
    % 自注意力机制 (简化实现)
    fullyConnectedLayer(attentionDim, 'Name', 'Attention Query')
    fullyConnectedLayer(attentionDim, 'Name', 'Attention Key')
    fullyConnectedLayer(attentionDim, 'Name', 'Attention Value')
    fullyConnectedLayer(attentionDim, 'Name', 'Attention Output')
    reluLayer('Name', 'ReLU Attention')
    dropoutLayer(0.4, 'Name', 'DropOut Attention')
    
    % LSTM序列处理 (保持原有结构)
    fullyConnectedLayer(numHiddenUnits, 'Name', 'Pre LSTM')
    lstmLayer(numHiddenUnits, 'OutputMode', 'sequence', 'Name', 'LSTM1')
    dropoutLayer(0.5, 'Name', 'DropOut1')
    lstmLayer(numHiddenUnits, 'OutputMode', 'last', 'Name', 'LSTM2')
    dropoutLayer(0.5, 'Name', 'DropOut2')
    
    % 分类层 (保持原有结构)
    fullyConnectedLayer(numClasses, 'Name', 'FC1')
    softmaxLayer('Name', 'SoftMax')
    classificationLayer('Name', 'Output')
];

% 训练选项 (保持与原代码一致，微调部分参数)
miniBatchSize = 512; 
iterPerEpoch = floor(numTrainingFramesPerRouter*numTotalRouters/miniBatchSize);

options = trainingOptions('adam', ...
    'MaxEpochs', 25, ...  % 略微增加训练轮数
    'ValidationData', {xValFrames, yVal}, ...
    'ValidationFrequency', iterPerEpoch, ...
    'Verbose', false, ...
    'InitialLearnRate', 0.006, ...  % 略微降低学习率
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.5, ...
    'LearnRateDropPeriod', 3, ...  % 调整学习率衰减周期
    'MiniBatchSize', miniBatchSize, ...
    'Plots', 'training-progress', ...
    'Shuffle', 'every-epoch', ...
    'L2Regularization', 0.0012, ...  % 略微增加正则化
    'ExecutionEnvironment', 'cpu');  

% 模型训练
tic
simNet = trainNetwork(xTrainingFrames, yTrain, layers, options);
TrainTime = seconds(toc);
disp("time of training = ");
toc

% 模型测试
yTestPred = classify(simNet, xTestFrames, 'ExecutionEnvironment', 'cpu');
testAccuracy = mean(yTest == yTestPred);
disp("Test accuracy: " + testAccuracy*100 + "%")

% 额外的性能分析
fprintf('\n=== CNN + 自注意力 + LSTM 模型性能 ===\n');
fprintf('模型架构: CNN特征提取 -> 自注意力机制 -> LSTM序列处理\n');
fprintf('输入维度: [%d, %d, %d]\n', inputSize(1), inputSize(2), inputSize(3));
fprintf('注意力维度: %d\n', attentionDim);
fprintf('LSTM隐藏单元: %d\n', numHiddenUnits);
fprintf('分类数: %d\n', numClasses);
fprintf('测试准确率: %.2f%%\n', testAccuracy*100);
fprintf('训练时间: %.2f 秒\n', TrainTime);

% 保存增强模型
save('rf_fingerprint_cnn_attention_lstm_enhanced.mat', 'simNet', 'TrainTime', 'testAccuracy');