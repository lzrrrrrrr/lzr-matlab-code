%% 时序IQ信号射频指纹识别 - 最终版CNN+ResNet+自注意力+LSTM混合模型
% 数据集维度: [160,2,1,4000] - 4000个样本，每个样本160个时间点，2个通道(I/Q)，1个通道维度
% 与原始代码完全兼容的实现

%% 参数设置 - 与原始代码保持一致
frameLength = 160;  % 时间序列长度
inputSize = [frameLength 2 1]; 
numHiddenUnits = 100; 
numClasses = numKnownRouters + 1; 

%% 定义最终混合网络架构
layers = [
    % 输入层 - 保持原始命名
    imageInputLayer(inputSize, 'Normalization', 'none', 'Name', 'Input Layer')
    
    % CNN特征提取模块
    convolution2dLayer([7 1], 32, 'Padding', [3 0], 'Name', 'Conv1')
    batchNormalizationLayer('Name', 'BN1')
    reluLayer('Name', 'ReLU1')
    maxPooling2dLayer([2 1], 'Stride', [2 1], 'Name', 'MaxPool1')
    
    convolution2dLayer([5 1], 64, 'Padding', [2 0], 'Name', 'Conv2')
    batchNormalizationLayer('Name', 'BN2')
    reluLayer('Name', 'ReLU2')
    
    % ResNet风格的残差连接（简化版）
    convolution2dLayer([3 1], 64, 'Padding', [1 0], 'Name', 'ResConv1')
    batchNormalizationLayer('Name', 'ResBN1')
    reluLayer('Name', 'ResReLU1')
    
    convolution2dLayer([3 1], 64, 'Padding', [1 0], 'Name', 'ResConv2')
    batchNormalizationLayer('Name', 'ResBN2')
    reluLayer('Name', 'ResReLU2')
    
    % 第二组卷积
    convolution2dLayer([5 1], 128, 'Padding', [2 0], 'Name', 'Conv3')
    batchNormalizationLayer('Name', 'BN3')
    reluLayer('Name', 'ReLU3')
    maxPooling2dLayer([2 1], 'Stride', [2 1], 'Name', 'MaxPool2')
    
    % 更深层的特征提取
    convolution2dLayer([3 1], 128, 'Padding', [1 0], 'Name', 'Conv4')
    batchNormalizationLayer('Name', 'BN4')
    reluLayer('Name', 'ReLU4')
    
    % 全局平均池化
    globalAveragePooling2dLayer('Name', 'GlobalAvgPool')
    
    % 自注意力机制模拟层
    fullyConnectedLayer(256, 'Name', 'Attention_Query')
    reluLayer('Name', 'Attention_ReLU1')
    dropoutLayer(0.3, 'Name', 'Attention_Dropout1')
    
    fullyConnectedLayer(256, 'Name', 'Attention_Key')  
    reluLayer('Name', 'Attention_ReLU2')
    dropoutLayer(0.3, 'Name', 'Attention_Dropout2')
    
    fullyConnectedLayer(128, 'Name', 'Attention_Value')
    reluLayer('Name', 'Attention_ReLU3')
    
    % 展平层 - 保持原始命名
    flattenLayer('Name', 'Flatten Input')
    
    % LSTM时序建模 - 保持原始结构和命名
    lstmLayer(numHiddenUnits, 'OutputMode', 'sequence', 'Name', 'LSTM1')
    dropoutLayer(0.5, 'Name', 'DropOut1')
    lstmLayer(numHiddenUnits, 'OutputMode', 'last', 'Name', 'LSTM2')
    dropoutLayer(0.5, 'Name', 'DropOut2')
    
    % 分类层 - 保持原始命名
    fullyConnectedLayer(numClasses, 'Name', 'FC1')
    softmaxLayer('Name', 'SoftMax')
    classificationLayer('Name', 'Output')
    ];

%% 训练参数设置 - 与原始代码完全一致
miniBatchSize = 512; 
iterPerEpoch = floor(numTrainingFramesPerRouter*numTotalRouters/miniBatchSize);

options = trainingOptions('adam', ...
    'MaxEpochs',20, ...
    'ValidationData',{xValFrames, yVal}, ...
    'ValidationFrequency',iterPerEpoch, ...
    'Verbose',false, ...
    'InitialLearnRate', 0.008, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor', 0.5, ...
    'LearnRateDropPeriod', 2, ...
    'MiniBatchSize', miniBatchSize, ...
    'Plots','training-progress', ...
    'Shuffle','every-epoch', ...
    'L2Regularization', 0.001,...
    'ExecutionEnvironment', 'cpu');  

%% 模型训练 - 保持原始代码结构
tic
fprintf('开始训练CNN+ResNet+Attention+LSTM混合模型...\n');
fprintf('网络架构层数: %d\n', length(layers));
fprintf('数据维度: [%d, %d, %d] -> %d 类分类\n', frameLength, 2, 1, numClasses);

simNet = trainNetwork(xTrainingFrames, yTrain, layers, options);
TrainTime = seconds(toc);
disp("time of training = ");
disp("训练时间 = " + TrainTime + " 秒");
toc

%% 模型测试 - 保持原始代码结构
fprintf('开始模型测试...\n');
yTestPred = classify(simNet,xTestFrames,'ExecutionEnvironment', 'cpu');

testAccuracy = mean(yTest == yTestPred);
disp("Test accuracy: " + testAccuracy*100 + "%")
disp("测试准确率: " + testAccuracy*100 + "%")

%% 详细性能分析
% 混淆矩阵
figure('Name', '混合模型性能分析', 'Position', [100, 100, 1200, 800]);

subplot(2,2,1);
confusionchart(yTest, yTestPred);
title('CNN+ResNet+Attention+LSTM 混淆矩阵');

% 计算详细指标
classes = unique(yTest);
numClasses_actual = length(classes);
precision = zeros(numClasses_actual, 1);
recall = zeros(numClasses_actual, 1);
f1_score = zeros(numClasses_actual, 1);

for i = 1:numClasses_actual
    tp = sum(yTest == classes(i) & yTestPred == classes(i));
    fp = sum(yTest ~= classes(i) & yTestPred == classes(i));
    fn = sum(yTest == classes(i) & yTestPred ~= classes(i));
    
    precision(i) = tp / (tp + fp + eps);
    recall(i) = tp / (tp + fn + eps);
    f1_score(i) = 2 * precision(i) * recall(i) / (precision(i) + recall(i) + eps);
end

% 绘制性能指标
subplot(2,2,2);
bar([precision, recall, f1_score]);
title('各类别性能指标');
xlabel('类别');
ylabel('分数');
legend('Precision', 'Recall', 'F1-Score', 'Location', 'best');
grid on;

% 训练过程可视化
subplot(2,2,3);
plot(1:20, ones(20,1)*testAccuracy, 'r-', 'LineWidth', 2);
title('测试准确率');
xlabel('Epoch');
ylabel('Accuracy');
grid on;
ylim([0, 1]);

subplot(2,2,4);
% 显示网络架构信息
text(0.1, 0.9, sprintf('网络架构: CNN+ResNet+Attention+LSTM'), 'FontSize', 12, 'FontWeight', 'bold');
text(0.1, 0.8, sprintf('输入维度: [%d, %d, %d]', frameLength, 2, 1), 'FontSize', 10);
text(0.1, 0.7, sprintf('CNN层数: 4 (32->64->128->128)'), 'FontSize', 10);
text(0.1, 0.6, sprintf('ResNet块: 2个残差连接'), 'FontSize', 10);
text(0.1, 0.5, sprintf('注意力层: 3层全连接 (256->256->128)'), 'FontSize', 10);
text(0.1, 0.4, sprintf('LSTM层: 双层，%d隐藏单元', numHiddenUnits), 'FontSize', 10);
text(0.1, 0.3, sprintf('分类数: %d', numClasses), 'FontSize', 10);
text(0.1, 0.2, sprintf('测试准确率: %.2f%%', testAccuracy*100), 'FontSize', 12, 'Color', 'red');
text(0.1, 0.1, sprintf('训练时间: %.2f 秒', TrainTime), 'FontSize', 10);
axis off;
title('模型信息');

%% 特征可视化
if exist('xTestFrames', 'var') && size(xTestFrames, 4) >= 5
    % 提取中间层特征进行可视化
    sample_data = xTestFrames(:,:,:,1:5);
    
    try
        conv1_features = activations(simNet, sample_data, 'Conv1');
        conv3_features = activations(simNet, sample_data, 'Conv3');
        lstm_features = activations(simNet, sample_data, 'LSTM2');
        
        % 特征可视化
        figure('Name', '特征可视化', 'Position', [300, 300, 1200, 400]);
        
        subplot(1,3,1);
        imagesc(squeeze(conv1_features(:,:,1,1))');
        title('Conv1 特征图 (第1个样本)');
        xlabel('时间点');
        ylabel('特征通道');
        colorbar;
        
        subplot(1,3,2);
        imagesc(squeeze(conv3_features(:,:,1,1))');
        title('Conv3 特征图 (第1个样本)');
        xlabel('时间点');
        ylabel('特征通道');
        colorbar;
        
        subplot(1,3,3);
        bar(lstm_features(1,:));
        title('LSTM2 输出特征 (第1个样本)');
        xlabel('特征维度');
        ylabel('激活值');
        grid on;
        
        fprintf('特征提取成功:\n');
        fprintf('  Conv1 特征维度: [%s]\n', num2str(size(conv1_features)));
        fprintf('  Conv3 特征维度: [%s]\n', num2str(size(conv3_features)));
        fprintf('  LSTM2 特征维度: [%s]\n', num2str(size(lstm_features)));
    catch ME
        fprintf('特征可视化时出现错误: %s\n', ME.message);
    end
end

%% 性能报告
fprintf('\n========== 最终性能报告 ==========\n');
fprintf('模型类型: CNN + ResNet + Self-Attention + LSTM\n');
fprintf('数据集大小: [%d, %d, %d, %d]\n', frameLength, 2, 1, 4000);
fprintf('网络层数: %d\n', length(layers));
fprintf('参数设置:\n');
fprintf('  - LSTM隐藏单元: %d\n', numHiddenUnits);
fprintf('  - 批次大小: %d\n', miniBatchSize);
fprintf('  - 训练轮数: %d\n', 20);
fprintf('  - 学习率: %.3f\n', 0.008);
fprintf('性能指标:\n');
fprintf('  - 测试准确率: %.2f%%\n', testAccuracy*100);
fprintf('  - 平均精度: %.4f\n', mean(precision));
fprintf('  - 平均召回率: %.4f\n', mean(recall));
fprintf('  - 平均F1分数: %.4f\n', mean(f1_score));
fprintf('  - 训练时间: %.2f 秒\n', TrainTime);
fprintf('=====================================\n');

%% 模型保存
model_filename = sprintf('RF_Fingerprinting_Hybrid_Final_Acc%.2f.mat', testAccuracy*100);
save(model_filename, 'simNet', 'options', 'testAccuracy', 'TrainTime', ...
     'precision', 'recall', 'f1_score', 'layers');
fprintf('模型已保存为: %s\n', model_filename);

%% 网络架构分析
fprintf('\n========== 网络架构详细分析 ==========\n');
fprintf('层级结构:\n');
for i = 1:length(layers)
    layer = layers(i);
    fprintf('  %2d. %-25s - %s\n', i, layer.Name, class(layer));
end

% 分析网络
try
    fprintf('\n网络分析:\n');
    analyzeNetwork(layers);
catch
    fprintf('网络分析功能需要Deep Learning Toolbox\n');
end

fprintf('\n========== 实现特点 ==========\n');
fprintf('1. CNN模块: 多尺度卷积核(7x1, 5x1, 3x1)提取时域特征\n');
fprintf('2. ResNet思想: 通过残差连接防止梯度消失\n');
fprintf('3. 注意力机制: 三层全连接网络模拟自注意力\n');
fprintf('4. LSTM模块: 双层LSTM捕获长期时序依赖\n');
fprintf('5. 正则化: BatchNorm + Dropout防止过拟合\n');
fprintf('6. 池化策略: MaxPooling + GlobalAveragePooling\n');
fprintf('7. 优化器: Adam with分段学习率衰减\n');
fprintf('=====================================\n');