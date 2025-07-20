%% 时序IQ信号射频指纹识别 - CNN+ResNet+自注意力+LSTM混合模型
% 数据集维度: [160,2,1,4000] - 4000个样本，每个样本160个时间点，2个通道(I/Q)，1个通道维度

%% 参数设置
frameLength = 160;  % 时间序列长度
numChannels = 2;    % I/Q两个通道
inputSize = [frameLength numChannels 1]; 
numHiddenUnits = 100; 
numClasses = numKnownRouters + 1; 

%% 定义混合网络架构
layers = [
    % 输入层
    imageInputLayer(inputSize, 'Normalization', 'none', 'Name', 'Input Layer')
    
    % CNN特征提取模块
    convolution2dLayer([5 1], 32, 'Padding', [2 0], 'Name', 'Conv1')
    batchNormalizationLayer('Name', 'BN1')
    reluLayer('Name', 'ReLU1')
    
    convolution2dLayer([5 1], 64, 'Padding', [2 0], 'Name', 'Conv2')
    batchNormalizationLayer('Name', 'BN2')
    reluLayer('Name', 'ReLU2')
    
    % ResNet残差模块1
    additionLayer(2, 'Name', 'Residual1')
    ];

% 创建ResNet残差连接的分支
residualBranch1 = [
    convolution2dLayer([3 1], 64, 'Padding', [1 0], 'Name', 'ResConv1_1')
    batchNormalizationLayer('Name', 'ResBN1_1')
    reluLayer('Name', 'ResReLU1_1')
    
    convolution2dLayer([3 1], 64, 'Padding', [1 0], 'Name', 'ResConv1_2')
    batchNormalizationLayer('Name', 'ResBN1_2')
    ];

% 主干网络继续
mainBranch = [
    % ResNet残差模块2
    reluLayer('Name', 'ReLU_Res1')
    convolution2dLayer([5 1], 128, 'Padding', [2 0], 'Name', 'Conv3')
    batchNormalizationLayer('Name', 'BN3')
    reluLayer('Name', 'ReLU3')
    
    additionLayer(2, 'Name', 'Residual2')
    ];

% 残差分支2
residualBranch2 = [
    convolution2dLayer([3 1], 128, 'Padding', [1 0], 'Name', 'ResConv2_1')
    batchNormalizationLayer('Name', 'ResBN2_1')
    reluLayer('Name', 'ResReLU2_1')
    
    convolution2dLayer([3 1], 128, 'Padding', [1 0], 'Name', 'ResConv2_2')
    batchNormalizationLayer('Name', 'ResBN2_2')
    ];

% 继续主干网络
finalLayers = [
    reluLayer('Name', 'ReLU_Res2')
    
    % 全局平均池化降维
    globalAveragePooling2dLayer('Name', 'GlobalAvgPool')
    
    % 重塑为序列数据准备LSTM输入
    flattenLayer('Name', 'Flatten_for_LSTM')
    
    % 自注意力机制模拟（通过全连接层实现）
    fullyConnectedLayer(256, 'Name', 'Attention_Query')
    reluLayer('Name', 'Attention_ReLU1')
    fullyConnectedLayer(256, 'Name', 'Attention_Key')
    reluLayer('Name', 'Attention_ReLU2')
    fullyConnectedLayer(128, 'Name', 'Attention_Value')
    reluLayer('Name', 'Attention_ReLU3')
    
    % LSTM时序建模
    lstmLayer(numHiddenUnits, 'OutputMode', 'sequence', 'Name', 'LSTM1')
    dropoutLayer(0.5, 'Name', 'DropOut1')
    lstmLayer(numHiddenUnits, 'OutputMode', 'last', 'Name', 'LSTM2')
    dropoutLayer(0.5, 'Name', 'DropOut2')
    
    % 分类层
    fullyConnectedLayer(numClasses, 'Name', 'FC1')
    softmaxLayer('Name', 'SoftMax')
    classificationLayer('Name', 'Output')
    ];

% 使用layerGraph创建复杂网络结构
lgraph = layerGraph();

% 添加主干层
lgraph = addLayers(lgraph, layers);
lgraph = addLayers(lgraph, mainBranch);
lgraph = addLayers(lgraph, finalLayers);

% 添加残差分支
lgraph = addLayers(lgraph, residualBranch1);
lgraph = addLayers(lgraph, residualBranch2);

% 连接残差分支1
lgraph = connectLayers(lgraph, 'ReLU2', 'ResConv1_1');
lgraph = connectLayers(lgraph, 'ResBN1_2', 'Residual1/in2');
lgraph = connectLayers(lgraph, 'ReLU2', 'Residual1/in1');

% 连接主干到残差模块1后的层
lgraph = connectLayers(lgraph, 'Residual1', 'ReLU_Res1');

% 连接残差分支2
lgraph = connectLayers(lgraph, 'ReLU3', 'ResConv2_1');
lgraph = connectLayers(lgraph, 'ResBN2_2', 'Residual2/in2');
lgraph = connectLayers(lgraph, 'ReLU3', 'Residual2/in1');

% 连接残差模块2后的层
lgraph = connectLayers(lgraph, 'Residual2', 'ReLU_Res2');

% 可视化网络结构
figure;
plot(lgraph);
title('CNN+ResNet+Attention+LSTM混合网络架构');

%% 训练参数设置
miniBatchSize = 512; 
iterPerEpoch = floor(numTrainingFramesPerRouter*numTotalRouters/miniBatchSize);

options = trainingOptions('adam', ...
    'MaxEpochs', 20, ...
    'ValidationData', {xValFrames, yVal}, ...
    'ValidationFrequency', iterPerEpoch, ...
    'Verbose', false, ...
    'InitialLearnRate', 0.008, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.5, ...
    'LearnRateDropPeriod', 2, ...
    'MiniBatchSize', miniBatchSize, ...
    'Plots', 'training-progress', ...
    'Shuffle', 'every-epoch', ...
    'L2Regularization', 0.001, ...
    'ExecutionEnvironment', 'cpu');

%% 模型训练
tic
fprintf('开始训练混合模型...\n');
simNet = trainNetwork(xTrainingFrames, yTrain, lgraph, options);
TrainTime = seconds(toc);
disp("训练时间 = " + TrainTime + " 秒");
toc

%% 模型测试
fprintf('开始模型测试...\n');
yTestPred = classify(simNet, xTestFrames, 'ExecutionEnvironment', 'cpu');

testAccuracy = mean(yTest == yTestPred);
disp("测试准确率: " + testAccuracy*100 + "%");

%% 结果分析
% 混淆矩阵
figure;
confusionchart(yTest, yTestPred);
title('混合模型混淆矩阵');

% 分类报告
fprintf('\n=== 分类性能报告 ===\n');
fprintf('测试准确率: %.2f%%\n', testAccuracy*100);
fprintf('训练时间: %.2f 秒\n', TrainTime);

%% 模型保存
save('RF_Fingerprinting_Hybrid_Model.mat', 'simNet', 'options', 'testAccuracy', 'TrainTime');
fprintf('模型已保存为 RF_Fingerprinting_Hybrid_Model.mat\n');

%% 特征可视化函数（可选）
function visualizeFeatures(net, inputData, layerName)
    % 可视化指定层的特征图
    act = activations(net, inputData, layerName);
    figure;
    montage(act(:,:,1:min(16,size(act,3)),1));
    title(['特征图 - ' layerName]);
end

%% 网络架构说明
fprintf('\n=== 网络架构说明 ===\n');
fprintf('1. CNN模块: 提取空间特征，使用5x1卷积核适应时序数据\n');
fprintf('2. ResNet模块: 两个残差块，防止梯度消失，加深网络\n');
fprintf('3. 注意力机制: 通过全连接层模拟，突出重要特征\n');
fprintf('4. LSTM模块: 双层LSTM处理时序依赖关系\n');
fprintf('5. 正则化: BatchNorm + Dropout防止过拟合\n');
fprintf('6. 数据流: [160,2,1] -> CNN -> ResNet -> Attention -> LSTM -> 分类\n');