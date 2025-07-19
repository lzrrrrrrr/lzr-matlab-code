%% 射频指纹识别：CNN + 自注意力机制 + LSTM 模型
% 数据集维度: [160, 2, 1, 4000] - 4000个样本，每个样本160个时间点，2个通道(I/Q)

%% 模型参数设置
frameLength = 160;  % 时间点数
inputSize = [frameLength 2 1]; 
numHiddenUnits = 100; 
numClasses = numKnownRouters + 1; 

%% 定义自注意力机制层
% 自注意力机制的自定义层
function layer = selfAttentionLayer(numHeads, keyDim, name)
    % 简化的自注意力机制实现
    % 使用全连接层模拟注意力计算
    layer = [
        fullyConnectedLayer(keyDim, 'Name', [name '_Query'])
        fullyConnectedLayer(keyDim, 'Name', [name '_Key']) 
        fullyConnectedLayer(keyDim, 'Name', [name '_Value'])
        % 注意力权重计算和应用将在后续层中实现
    ];
end

%% 构建CNN + 自注意力 + LSTM混合模型
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
    
    % 展平层，为LSTM准备序列数据
    flattenLayer('Name', 'Flatten CNN Output')
    
    % 特征维度调整层
    fullyConnectedLayer(256, 'Name', 'Feature Projection')
    reluLayer('Name', 'ReLU Feature')
    dropoutLayer(0.3, 'Name', 'DropOut Feature')
    
    % 重塑为序列格式 (这里用全连接层模拟自注意力机制)
    % 自注意力机制模拟 - Query, Key, Value计算
    fullyConnectedLayer(256, 'Name', 'Attention Query')
    fullyConnectedLayer(256, 'Name', 'Attention Key')  
    fullyConnectedLayer(256, 'Name', 'Attention Value')
    
    % 注意力权重应用 (简化实现)
    fullyConnectedLayer(256, 'Name', 'Attention Output')
    reluLayer('Name', 'ReLU Attention')
    dropoutLayer(0.4, 'Name', 'DropOut Attention')
    
    % LSTM序列处理部分
    % 需要将数据重新整形为序列格式
    fullyConnectedLayer(numHiddenUnits, 'Name', 'Pre LSTM FC')
    
    % 第一个LSTM层
    lstmLayer(numHiddenUnits, 'OutputMode', 'sequence', 'Name', 'LSTM1')
    dropoutLayer(0.5, 'Name', 'DropOut1')
    
    % 第二个LSTM层
    lstmLayer(numHiddenUnits, 'OutputMode', 'last', 'Name', 'LSTM2')
    dropoutLayer(0.5, 'Name', 'DropOut2')
    
    % 分类层
    fullyConnectedLayer(numClasses, 'Name', 'FC1')
    softmaxLayer('Name', 'SoftMax')
    classificationLayer('Name', 'Output')
];

%% 训练选项设置
miniBatchSize = 512; 
iterPerEpoch = floor(numTrainingFramesPerRouter*numTotalRouters/miniBatchSize);

options = trainingOptions('adam', ...
    'MaxEpochs', 25, ...  % 增加训练轮数以适应更复杂的模型
    'ValidationData', {xValFrames, yVal}, ...
    'ValidationFrequency', iterPerEpoch, ...
    'Verbose', false, ...
    'InitialLearnRate', 0.005, ...  % 降低学习率以适应复杂模型
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.5, ...
    'LearnRateDropPeriod', 3, ...  % 调整学习率下降周期
    'MiniBatchSize', miniBatchSize, ...
    'Plots', 'training-progress', ...
    'Shuffle', 'every-epoch', ...
    'L2Regularization', 0.0015, ...  % 增加正则化以防止过拟合
    'ExecutionEnvironment', 'cpu');

%% 模型训练
fprintf('开始训练CNN + 自注意力 + LSTM混合模型...\n');
tic
simNet = trainNetwork(xTrainingFrames, yTrain, layers, options);
TrainTime = seconds(toc);
disp("训练时间 = ");
toc

%% 模型测试
fprintf('开始模型测试...\n');
yTestPred = classify(simNet, xTestFrames, 'ExecutionEnvironment', 'cpu');

testAccuracy = mean(yTest == yTestPred);
disp("测试准确率: " + testAccuracy*100 + "%")

%% 结果分析和可视化
% 混淆矩阵
figure;
confusionchart(yTest, yTestPred);
title('CNN + 自注意力 + LSTM 模型混淆矩阵');

% 准确率对比
fprintf('\n=== 模型性能总结 ===\n');
fprintf('测试准确率: %.2f%%\n', testAccuracy*100);
fprintf('训练时间: %.2f 秒\n', TrainTime);

%% 模型复杂度分析
fprintf('\n=== 模型架构信息 ===\n');
fprintf('输入维度: [%d, %d, %d]\n', inputSize(1), inputSize(2), inputSize(3));
fprintf('隐藏单元数: %d\n', numHiddenUnits);
fprintf('分类数: %d\n', numClasses);
fprintf('批次大小: %d\n', miniBatchSize);

% 保存训练好的模型
save('rf_fingerprint_cnn_attention_lstm_model.mat', 'simNet', 'TrainTime', 'testAccuracy');
fprintf('模型已保存为: rf_fingerprint_cnn_attention_lstm_model.mat\n');