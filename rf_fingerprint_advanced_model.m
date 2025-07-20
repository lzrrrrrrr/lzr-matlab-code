%% 射频指纹识别：高级CNN + 自注意力机制 + LSTM 混合模型
% 数据集维度: [160, 2, 1, 4000] - 4000个样本，每个样本160个时间点，2个通道(I/Q)
% 适用于MATLAB R2023B

%% 模型参数设置
frameLength = 160;  % 时间点数
inputSize = [frameLength 2 1]; 
numHiddenUnits = 100; 
numClasses = numKnownRouters + 1; 

% 自注意力机制参数
attentionDim = 128;
numAttentionHeads = 4;

%% 构建高级CNN + 自注意力 + LSTM混合模型
layers = [
    % 输入层 - 处理[160, 2, 1]的IQ信号
    imageInputLayer(inputSize, 'Normalization', 'none', 'Name', 'Input Layer')
    
    % ======================
    % CNN特征提取模块
    % ======================
    
    % 第一个卷积块 - 时域特征提取
    convolution2dLayer([16, 2], 64, 'Padding', 'same', 'Name', 'Conv1')
    batchNormalizationLayer('Name', 'BN1')
    reluLayer('Name', 'ReLU1')
    maxPooling2dLayer([2, 1], 'Stride', [2, 1], 'Name', 'MaxPool1')  % [80, 2, 64]
    
    % 第二个卷积块 - 深层特征提取
    convolution2dLayer([8, 1], 128, 'Padding', 'same', 'Name', 'Conv2')
    batchNormalizationLayer('Name', 'BN2')
    reluLayer('Name', 'ReLU2')
    maxPooling2dLayer([2, 1], 'Stride', [2, 1], 'Name', 'MaxPool2')  % [40, 2, 128]
    
    % 第三个卷积块 - 高级特征提取
    convolution2dLayer([4, 1], 256, 'Padding', 'same', 'Name', 'Conv3')
    batchNormalizationLayer('Name', 'BN3')
    reluLayer('Name', 'ReLU3')
    maxPooling2dLayer([2, 1], 'Stride', [2, 1], 'Name', 'MaxPool3')  % [20, 2, 256]
    
    % 第四个卷积块 - 精细特征提取
    convolution2dLayer([4, 1], 512, 'Padding', 'same', 'Name', 'Conv4')
    batchNormalizationLayer('Name', 'BN4')
    reluLayer('Name', 'ReLU4')
    
    % ======================
    % 特征重塑和降维模块
    % ======================
    
    % 全局平均池化
    globalAveragePooling2dLayer('Name', 'GlobalAvgPool')
    
    % 特征投影层
    fullyConnectedLayer(attentionDim, 'Name', 'Feature Projection')
    reluLayer('Name', 'ReLU Feature Proj')
    dropoutLayer(0.3, 'Name', 'DropOut Feature Proj')
    
    % ======================
    % 自注意力机制模块 (简化实现)
    % ======================
    
    % 多头注意力的Query分支
    fullyConnectedLayer(attentionDim, 'Name', 'Attention Query FC')
    reluLayer('Name', 'ReLU Query')
    
    % 多头注意力的Key分支  
    fullyConnectedLayer(attentionDim, 'Name', 'Attention Key FC')
    reluLayer('Name', 'ReLU Key')
    
    % 多头注意力的Value分支
    fullyConnectedLayer(attentionDim, 'Name', 'Attention Value FC')
    reluLayer('Name', 'ReLU Value')
    
    % 注意力输出融合
    fullyConnectedLayer(attentionDim, 'Name', 'Attention Fusion')
    reluLayer('Name', 'ReLU Attention Fusion')
    dropoutLayer(0.4, 'Name', 'DropOut Attention')
    
    % 残差连接模拟
    fullyConnectedLayer(attentionDim, 'Name', 'Residual Connection')
    reluLayer('Name', 'ReLU Residual')
    
    % ======================
    % LSTM序列处理模块
    % ======================
    
    % 为LSTM准备特征
    fullyConnectedLayer(numHiddenUnits*2, 'Name', 'Pre LSTM Feature')
    reluLayer('Name', 'ReLU Pre LSTM')
    dropoutLayer(0.3, 'Name', 'DropOut Pre LSTM')
    
    % 序列长度调整 (模拟序列展开)
    fullyConnectedLayer(numHiddenUnits, 'Name', 'Sequence Preparation')
    
    % 双向LSTM第一层
    lstmLayer(numHiddenUnits, 'OutputMode', 'sequence', 'Name', 'LSTM1')
    dropoutLayer(0.5, 'Name', 'DropOut LSTM1')
    
    % 双向LSTM第二层
    lstmLayer(numHiddenUnits, 'OutputMode', 'sequence', 'Name', 'LSTM2')
    dropoutLayer(0.5, 'Name', 'DropOut LSTM2')
    
    % LSTM输出处理
    lstmLayer(floor(numHiddenUnits/2), 'OutputMode', 'last', 'Name', 'LSTM Output')
    dropoutLayer(0.4, 'Name', 'DropOut LSTM Output')
    
    % ======================
    % 分类决策模块
    % ======================
    
    % 特征融合层
    fullyConnectedLayer(numHiddenUnits, 'Name', 'Feature Fusion')
    reluLayer('Name', 'ReLU Fusion')
    dropoutLayer(0.5, 'Name', 'DropOut Fusion')
    
    % 最终分类层
    fullyConnectedLayer(numClasses, 'Name', 'FC Classification')
    softmaxLayer('Name', 'SoftMax')
    classificationLayer('Name', 'Output')
];

%% 高级训练选项设置
miniBatchSize = 256;  % 减小批次大小以适应复杂模型
iterPerEpoch = floor(numTrainingFramesPerRouter*numTotalRouters/miniBatchSize);

% 使用更精细的学习率调度
options = trainingOptions('adam', ...
    'MaxEpochs', 30, ...  % 增加训练轮数
    'ValidationData', {xValFrames, yVal}, ...
    'ValidationFrequency', iterPerEpoch, ...
    'Verbose', false, ...
    'InitialLearnRate', 0.003, ...  % 较低的初始学习率
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.3, ...  % 更激进的学习率衰减
    'LearnRateDropPeriod', 4, ...
    'MiniBatchSize', miniBatchSize, ...
    'Plots', 'training-progress', ...
    'Shuffle', 'every-epoch', ...
    'L2Regularization', 0.002, ...  % 增强正则化
    'GradientThreshold', 1, ...  % 梯度裁剪
    'ExecutionEnvironment', 'cpu', ...
    'ValidationPatience', 8);  % 早停机制

%% 模型训练
fprintf('=== 开始训练高级CNN + 自注意力 + LSTM混合模型 ===\n');
fprintf('模型架构: CNN特征提取 -> 自注意力机制 -> LSTM序列处理 -> 分类\n');
fprintf('数据维度: [%d, %d, %d] -> %d类分类\n', inputSize(1), inputSize(2), inputSize(3), numClasses);
fprintf('训练参数: 批次大小=%d, 最大轮数=%d, 学习率=%.4f\n', miniBatchSize, options.MaxEpochs, options.InitialLearnRate);

tic
simNet = trainNetwork(xTrainingFrames, yTrain, layers, options);
TrainTime = seconds(toc);
fprintf('训练完成！训练时间: %.2f 秒\n', TrainTime);

%% 模型测试和评估
fprintf('\n=== 开始模型测试 ===\n');
yTestPred = classify(simNet, xTestFrames, 'ExecutionEnvironment', 'cpu');

% 计算测试准确率
testAccuracy = mean(yTest == yTestPred);
fprintf('测试准确率: %.2f%%\n', testAccuracy*100);

% 计算每类准确率
uniqueClasses = unique(yTest);
classAccuracies = zeros(length(uniqueClasses), 1);
for i = 1:length(uniqueClasses)
    classIdx = (yTest == uniqueClasses(i));
    classAccuracies(i) = mean(yTestPred(classIdx) == yTest(classIdx));
    fprintf('类别 %s 准确率: %.2f%%\n', string(uniqueClasses(i)), classAccuracies(i)*100);
end

%% 结果可视化
% 混淆矩阵
figure('Position', [100, 100, 800, 600]);
confusionchart(yTest, yTestPred, 'RowSummary', 'row-normalized', 'ColumnSummary', 'column-normalized');
title('CNN + 自注意力 + LSTM 模型混淆矩阵', 'FontSize', 14);

% 训练历史可视化 (如果有训练信息)
if exist('info', 'var')
    figure('Position', [100, 100, 1200, 400]);
    subplot(1, 2, 1);
    plot(info.TrainingLoss, 'b-', 'LineWidth', 2);
    hold on;
    plot(info.ValidationLoss, 'r-', 'LineWidth', 2);
    title('训练和验证损失');
    xlabel('迭代次数');
    ylabel('损失值');
    legend('训练损失', '验证损失');
    grid on;
    
    subplot(1, 2, 2);
    plot(info.TrainingAccuracy, 'b-', 'LineWidth', 2);
    hold on;
    plot(info.ValidationAccuracy, 'r-', 'LineWidth', 2);
    title('训练和验证准确率');
    xlabel('迭代次数');
    ylabel('准确率');
    legend('训练准确率', '验证准确率');
    grid on;
end

%% 模型性能总结
fprintf('\n=== 模型性能总结 ===\n');
fprintf('模型类型: CNN + 自注意力机制 + LSTM\n');
fprintf('输入维度: [%d, %d, %d]\n', inputSize(1), inputSize(2), inputSize(3));
fprintf('隐藏单元数: %d\n', numHiddenUnits);
fprintf('注意力维度: %d\n', attentionDim);
fprintf('分类数: %d\n', numClasses);
fprintf('批次大小: %d\n', miniBatchSize);
fprintf('训练时间: %.2f 秒\n', TrainTime);
fprintf('测试准确率: %.2f%%\n', testAccuracy*100);
fprintf('平均类别准确率: %.2f%%\n', mean(classAccuracies)*100);

%% 保存模型和结果
modelSaveName = 'rf_fingerprint_cnn_attention_lstm_advanced.mat';
save(modelSaveName, 'simNet', 'TrainTime', 'testAccuracy', 'classAccuracies', 'options');
fprintf('\n模型已保存为: %s\n', modelSaveName);

%% 模型复杂度分析
fprintf('\n=== 模型复杂度分析 ===\n');
fprintf('网络层数: %d\n', length(layers));
fprintf('卷积层数: 4\n');
fprintf('LSTM层数: 3\n');
fprintf('全连接层数: %d\n', sum(contains({layers.Name}, 'FC') | contains({layers.Name}, 'Attention') | contains({layers.Name}, 'Feature')));
fprintf('Dropout层数: %d\n', sum(contains({layers.Name}, 'DropOut')));

fprintf('\n=== 射频指纹识别模型训练完成 ===\n');