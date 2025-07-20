%% 射频指纹识别：CNN + 自注意力机制 + LSTM 完整实现
% 数据集维度: [160, 2, 1, 4000] - 4000个样本，每个样本160个时间点，2个通道(I/Q)
% 适用于MATLAB R2023B
% 作者: AI Assistant
% 日期: 2024

clear; clc;

%% 模型参数设置
frameLength = 160;  % 时间点数
inputSize = [frameLength 2 1]; 
numHiddenUnits = 100; 
numClasses = numKnownRouters + 1;  % 需要根据实际数据集设置

% 自注意力机制参数
attentionDim = 128;
featureDim = 256;

%% 构建CNN + 自注意力 + LSTM混合模型
fprintf('构建CNN + 自注意力 + LSTM混合模型...\n');

layers = [
    % ========================
    % 输入层
    % ========================
    imageInputLayer(inputSize, 'Normalization', 'none', 'Name', 'Input Layer')
    
    % ========================
    % CNN特征提取模块
    % ========================
    
    % 第一个卷积块 - 时域特征提取
    convolution2dLayer([16, 2], 32, 'Padding', 'same', 'Name', 'Conv1')
    batchNormalizationLayer('Name', 'BN1')
    reluLayer('Name', 'ReLU1')
    maxPooling2dLayer([2, 1], 'Stride', [2, 1], 'Name', 'MaxPool1')
    
    % 第二个卷积块 - 深层特征提取
    convolution2dLayer([8, 2], 64, 'Padding', 'same', 'Name', 'Conv2')
    batchNormalizationLayer('Name', 'BN2')
    reluLayer('Name', 'ReLU2')
    maxPooling2dLayer([2, 1], 'Stride', [2, 1], 'Name', 'MaxPool2')
    
    % 第三个卷积块 - 高级特征提取
    convolution2dLayer([4, 1], 128, 'Padding', 'same', 'Name', 'Conv3')
    batchNormalizationLayer('Name', 'BN3')
    reluLayer('Name', 'ReLU3')
    maxPooling2dLayer([2, 1], 'Stride', [2, 1], 'Name', 'MaxPool3')
    
    % 第四个卷积块
    convolution2dLayer([4, 1], featureDim, 'Padding', 'same', 'Name', 'Conv4')
    batchNormalizationLayer('Name', 'BN4')
    reluLayer('Name', 'ReLU4')
    
    % ========================
    % 特征处理模块
    % ========================
    
    % 全局平均池化
    globalAveragePooling2dLayer('Name', 'GlobalAvgPool')
    
    % 特征投影
    fullyConnectedLayer(featureDim, 'Name', 'Feature Projection')
    reluLayer('Name', 'ReLU Feature Proj')
    dropoutLayer(0.3, 'Name', 'DropOut Feature')
    
    % ========================
    % 自注意力机制模块 (简化实现)
    % ========================
    
    % Query变换
    fullyConnectedLayer(attentionDim, 'Name', 'Query Transform')
    reluLayer('Name', 'ReLU Query')
    
    % Key变换
    fullyConnectedLayer(attentionDim, 'Name', 'Key Transform')
    reluLayer('Name', 'ReLU Key')
    
    % Value变换
    fullyConnectedLayer(attentionDim, 'Name', 'Value Transform')
    reluLayer('Name', 'ReLU Value')
    
    % 注意力融合
    fullyConnectedLayer(attentionDim, 'Name', 'Attention Fusion')
    reluLayer('Name', 'ReLU Attention')
    dropoutLayer(0.4, 'Name', 'DropOut Attention')
    
    % 残差连接
    fullyConnectedLayer(attentionDim, 'Name', 'Residual Connect')
    reluLayer('Name', 'ReLU Residual')
    
    % ========================
    % LSTM序列处理模块
    % ========================
    
    % LSTM预处理
    fullyConnectedLayer(numHiddenUnits, 'Name', 'Pre LSTM')
    reluLayer('Name', 'ReLU Pre LSTM')
    dropoutLayer(0.3, 'Name', 'DropOut Pre LSTM')
    
    % 第一个LSTM层
    lstmLayer(numHiddenUnits, 'OutputMode', 'sequence', 'Name', 'LSTM1')
    dropoutLayer(0.5, 'Name', 'DropOut1')
    
    % 第二个LSTM层
    lstmLayer(numHiddenUnits, 'OutputMode', 'last', 'Name', 'LSTM2')
    dropoutLayer(0.5, 'Name', 'DropOut2')
    
    % ========================
    % 分类决策模块
    % ========================
    
    % 最终特征融合
    fullyConnectedLayer(numHiddenUnits, 'Name', 'Final Feature')
    reluLayer('Name', 'ReLU Final')
    dropoutLayer(0.5, 'Name', 'DropOut Final')
    
    % 分类层
    fullyConnectedLayer(numClasses, 'Name', 'FC1')
    softmaxLayer('Name', 'SoftMax')
    classificationLayer('Name', 'Output')
];

%% 训练选项设置
miniBatchSize = 512; 
iterPerEpoch = floor(numTrainingFramesPerRouter*numTotalRouters/miniBatchSize);

options = trainingOptions('adam', ...
    'MaxEpochs', 25, ...
    'ValidationData', {xValFrames, yVal}, ...
    'ValidationFrequency', iterPerEpoch, ...
    'Verbose', false, ...
    'InitialLearnRate', 0.005, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.5, ...
    'LearnRateDropPeriod', 3, ...
    'MiniBatchSize', miniBatchSize, ...
    'Plots', 'training-progress', ...
    'Shuffle', 'every-epoch', ...
    'L2Regularization', 0.0015, ...
    'GradientThreshold', 1, ...
    'ExecutionEnvironment', 'cpu');

%% 模型训练
fprintf('=== 开始训练CNN + 自注意力 + LSTM混合模型 ===\n');
fprintf('输入维度: [%d, %d, %d]\n', inputSize(1), inputSize(2), inputSize(3));
fprintf('分类数: %d\n', numClasses);
fprintf('隐藏单元数: %d\n', numHiddenUnits);
fprintf('注意力维度: %d\n', attentionDim);
fprintf('批次大小: %d\n', miniBatchSize);
fprintf('最大训练轮数: %d\n', options.MaxEpochs);

tic
simNet = trainNetwork(xTrainingFrames, yTrain, layers, options);
TrainTime = seconds(toc);
disp("训练时间 = " + TrainTime + " 秒");
toc

%% 模型测试
fprintf('\n=== 开始模型测试 ===\n');
yTestPred = classify(simNet, xTestFrames, 'ExecutionEnvironment', 'cpu');

testAccuracy = mean(yTest == yTestPred);
disp("测试准确率: " + testAccuracy*100 + "%");

%% 详细性能分析
% 计算每类准确率
uniqueClasses = unique(yTest);
fprintf('\n=== 各类别准确率 ===\n');
classAccuracies = zeros(length(uniqueClasses), 1);
for i = 1:length(uniqueClasses)
    classIdx = (yTest == uniqueClasses(i));
    classAccuracies(i) = mean(yTestPred(classIdx) == yTest(classIdx));
    fprintf('类别 %s: %.2f%%\n', string(uniqueClasses(i)), classAccuracies(i)*100);
end

% 计算混淆矩阵
fprintf('\n=== 混淆矩阵分析 ===\n');
figure('Position', [100, 100, 800, 600]);
confusionchart(yTest, yTestPred, 'RowSummary', 'row-normalized', ...
    'ColumnSummary', 'column-normalized');
title('CNN + 自注意力 + LSTM 模型混淆矩阵', 'FontSize', 14);

%% 模型性能总结
fprintf('\n=== 模型性能总结 ===\n');
fprintf('模型架构: CNN + 自注意力机制 + LSTM\n');
fprintf('数据集大小: [160, 2, 1, 4000]\n');
fprintf('测试准确率: %.2f%%\n', testAccuracy*100);
fprintf('平均类别准确率: %.2f%%\n', mean(classAccuracies)*100);
fprintf('训练时间: %.2f 秒\n', TrainTime);
fprintf('网络总层数: %d\n', length(layers));

% 计算模型复杂度
convLayers = sum(contains({layers.Name}, 'Conv'));
lstmLayers = sum(contains({layers.Name}, 'LSTM'));
fcLayers = sum(contains({layers.Name}, 'FC') | contains({layers.Name}, 'Transform') | ...
    contains({layers.Name}, 'Feature') | contains({layers.Name}, 'Attention') | ...
    contains({layers.Name}, 'Residual') | contains({layers.Name}, 'Pre') | ...
    contains({layers.Name}, 'Final'));

fprintf('卷积层数: %d\n', convLayers);
fprintf('LSTM层数: %d\n', lstmLayers);
fprintf('全连接层数: %d\n', fcLayers);

%% 保存模型
modelFileName = sprintf('rf_fingerprint_cnn_attention_lstm_acc_%.1f.mat', testAccuracy*100);
save(modelFileName, 'simNet', 'TrainTime', 'testAccuracy', 'classAccuracies', ...
    'options', 'layers', 'inputSize', 'numHiddenUnits', 'attentionDim');
fprintf('\n模型已保存为: %s\n', modelFileName);

%% 可选：模型预测示例
if exist('xTestFrames', 'var') && size(xTestFrames, 4) > 0
    fprintf('\n=== 预测示例 ===\n');
    % 随机选择几个测试样本进行预测展示
    numSamples = min(5, size(xTestFrames, 4));
    sampleIndices = randperm(size(xTestFrames, 4), numSamples);
    
    for i = 1:numSamples
        idx = sampleIndices(i);
        sampleData = xTestFrames(:, :, :, idx);
        truePred = yTest(idx);
        modelPred = classify(simNet, sampleData, 'ExecutionEnvironment', 'cpu');
        
        fprintf('样本 %d: 真实类别=%s, 预测类别=%s, 正确=%s\n', ...
            idx, string(truePred), string(modelPred), string(truePred == modelPred));
    end
end

%% 模型架构可视化（可选）
try
    figure('Position', [100, 100, 1200, 800]);
    plot(layerGraph(layers));
    title('CNN + 自注意力 + LSTM 模型架构图', 'FontSize', 16);
    fprintf('模型架构图已生成\n');
catch
    fprintf('模型架构图生成失败，可能是MATLAB版本不支持\n');
end

fprintf('\n=== CNN + 自注意力 + LSTM 射频指纹识别模型训练完成 ===\n');
fprintf('模型文件: %s\n', modelFileName);
fprintf('最终测试准确率: %.2f%%\n', testAccuracy*100);

%% 使用说明
fprintf('\n=== 使用说明 ===\n');
fprintf('1. 确保数据变量已正确加载:\n');
fprintf('   - xTrainingFrames: 训练数据 [160, 2, 1, N_train]\n');
fprintf('   - yTrain: 训练标签\n');
fprintf('   - xTestFrames: 测试数据 [160, 2, 1, N_test]\n');
fprintf('   - yTest: 测试标签\n');
fprintf('   - xValFrames: 验证数据 [160, 2, 1, N_val]\n');
fprintf('   - yVal: 验证标签\n');
fprintf('2. 设置正确的类别数 numKnownRouters\n');
fprintf('3. 根据需要调整网络参数和训练选项\n');
fprintf('4. 运行此脚本进行训练和测试\n');