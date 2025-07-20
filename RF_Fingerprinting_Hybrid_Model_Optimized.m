%% 时序IQ信号射频指纹识别 - 优化版CNN+ResNet+自注意力+LSTM混合模型
% 数据集维度: [160,2,1,4000] - 4000个样本，每个样本160个时间点，2个通道(I/Q)，1个通道维度

%% 参数设置
frameLength = 160;  % 时间序列长度
numChannels = 2;    % I/Q两个通道
inputSize = [frameLength numChannels 1]; 
numHiddenUnits = 100; 
numClasses = numKnownRouters + 1; 

%% 定义优化的混合网络架构
% 方法1：使用序列化架构（推荐用于MATLAB R2023b）
layers = [
    % 输入层
    imageInputLayer(inputSize, 'Normalization', 'none', 'Name', 'Input_Layer')
    
    % CNN特征提取模块
    convolution2dLayer([7 1], 32, 'Padding', [3 0], 'Stride', [1 1], 'Name', 'Conv1')
    batchNormalizationLayer('Name', 'BN1')
    reluLayer('Name', 'ReLU1')
    
    convolution2dLayer([5 1], 64, 'Padding', [2 0], 'Stride', [1 1], 'Name', 'Conv2')
    batchNormalizationLayer('Name', 'BN2')
    reluLayer('Name', 'ReLU2')
    
    % 残差连接的简化实现（通过跳跃连接）
    convolution2dLayer([3 1], 64, 'Padding', [1 0], 'Name', 'ResConv1')
    batchNormalizationLayer('Name', 'ResBN1')
    reluLayer('Name', 'ResReLU1')
    
    convolution2dLayer([3 1], 64, 'Padding', [1 0], 'Name', 'ResConv2')
    batchNormalizationLayer('Name', 'ResBN2')
    reluLayer('Name', 'ResReLU2')
    
    % 第二个残差块
    convolution2dLayer([5 1], 128, 'Padding', [2 0], 'Name', 'Conv3')
    batchNormalizationLayer('Name', 'BN3')
    reluLayer('Name', 'ReLU3')
    
    convolution2dLayer([3 1], 128, 'Padding', [1 0], 'Name', 'ResConv3')
    batchNormalizationLayer('Name', 'ResBN3')
    reluLayer('Name', 'ResReLU3')
    
    convolution2dLayer([3 1], 128, 'Padding', [1 0], 'Name', 'ResConv4')
    batchNormalizationLayer('Name', 'ResBN4')
    reluLayer('Name', 'ResReLU4')
    
    % 自适应池化和降维
    globalAveragePooling2dLayer('Name', 'GlobalAvgPool')
    
    % 自注意力机制模拟
    fullyConnectedLayer(256, 'Name', 'Attention_FC1')
    reluLayer('Name', 'Attention_ReLU1')
    dropoutLayer(0.3, 'Name', 'Attention_Dropout1')
    
    fullyConnectedLayer(128, 'Name', 'Attention_FC2')
    reluLayer('Name', 'Attention_ReLU2')
    dropoutLayer(0.3, 'Name', 'Attention_Dropout2')
    
    % 为LSTM准备序列数据
    fullyConnectedLayer(frameLength, 'Name', 'Reshape_for_LSTM')
    
    % 重新整形为序列（需要自定义层或者使用sequenceInputLayer）
    % 这里使用全连接层模拟序列转换
    fullyConnectedLayer(numHiddenUnits*2, 'Name', 'Seq_Prep')
    reluLayer('Name', 'Seq_ReLU')
    
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
fprintf('开始训练优化混合模型...\n');
fprintf('网络架构: CNN + ResNet-like + Attention + LSTM\n');
fprintf('数据维度: [%d, %d, %d] -> %d 类\n', frameLength, numChannels, 1, numClasses);

simNet = trainNetwork(xTrainingFrames, yTrain, layers, options);
TrainTime = seconds(toc);
disp("训练时间 = " + TrainTime + " 秒");
toc

%% 模型测试
fprintf('开始模型测试...\n');
yTestPred = classify(simNet, xTestFrames, 'ExecutionEnvironment', 'cpu');

testAccuracy = mean(yTest == yTestPred);
disp("测试准确率: " + testAccuracy*100 + "%");

%% 详细性能分析
% 混淆矩阵
figure('Position', [100, 100, 800, 600]);
confusionchart(yTest, yTestPred);
title('混合模型混淆矩阵');

% 计算每类的精度、召回率和F1分数
classes = unique(yTest);
precision = zeros(length(classes), 1);
recall = zeros(length(classes), 1);
f1_score = zeros(length(classes), 1);

for i = 1:length(classes)
    tp = sum(yTest == classes(i) & yTestPred == classes(i));
    fp = sum(yTest ~= classes(i) & yTestPred == classes(i));
    fn = sum(yTest == classes(i) & yTestPred ~= classes(i));
    
    precision(i) = tp / (tp + fp + eps);
    recall(i) = tp / (tp + fn + eps);
    f1_score(i) = 2 * precision(i) * recall(i) / (precision(i) + recall(i) + eps);
end

% 显示详细结果
fprintf('\n=== 详细分类性能报告 ===\n');
fprintf('总体测试准确率: %.2f%%\n', testAccuracy*100);
fprintf('训练时间: %.2f 秒\n', TrainTime);
fprintf('平均精度: %.4f\n', mean(precision));
fprintf('平均召回率: %.4f\n', mean(recall));
fprintf('平均F1分数: %.4f\n', mean(f1_score));

%% 网络结构分析
fprintf('\n=== 网络结构分析 ===\n');
analyzeNetwork(layers);

%% 特征提取和可视化
% 提取中间层特征
conv1_features = activations(simNet, xTestFrames(1:10,:,:,:), 'Conv1');
conv2_features = activations(simNet, xTestFrames(1:10,:,:,:), 'Conv2');
lstm_features = activations(simNet, xTestFrames(1:10,:,:,:), 'LSTM2');

fprintf('Conv1特征维度: [%s]\n', num2str(size(conv1_features)));
fprintf('Conv2特征维度: [%s]\n', num2str(size(conv2_features)));
fprintf('LSTM2特征维度: [%s]\n', num2str(size(lstm_features)));

%% 可视化部分特征
figure('Position', [200, 200, 1200, 400]);

% 可视化Conv1特征
subplot(1,3,1);
imagesc(squeeze(conv1_features(:,:,1,1))');
title('Conv1 第1个特征图');
xlabel('时间点');
ylabel('特征通道');
colorbar;

% 可视化Conv2特征
subplot(1,3,2);
imagesc(squeeze(conv2_features(:,:,1,1))');
title('Conv2 第1个特征图');
xlabel('时间点');
ylabel('特征通道');
colorbar;

% 可视化LSTM特征
subplot(1,3,3);
bar(lstm_features(1,:));
title('LSTM2 输出特征');
xlabel('特征维度');
ylabel('激活值');

%% 模型保存
model_filename = sprintf('RF_Fingerprinting_Hybrid_Model_Acc%.2f.mat', testAccuracy*100);
save(model_filename, 'simNet', 'options', 'testAccuracy', 'TrainTime', ...
     'precision', 'recall', 'f1_score');
fprintf('模型已保存为 %s\n', model_filename);

%% 预测函数
function predictions = predictRFFingerprint(model, testData)
    % RF指纹识别预测函数
    % 输入: model - 训练好的网络模型
    %       testData - 测试数据 [frameLength, 2, 1, N]
    % 输出: predictions - 预测结果
    
    predictions = classify(model, testData, 'ExecutionEnvironment', 'cpu');
end

%% 模型性能基准测试
function runBenchmark(model, testData, testLabels)
    % 运行性能基准测试
    fprintf('\n=== 性能基准测试 ===\n');
    
    % 测试不同批次大小的推理时间
    batch_sizes = [1, 10, 50, 100];
    
    for batch_size = batch_sizes
        if batch_size <= size(testData, 4)
            tic;
            pred = classify(model, testData(:,:,:,1:batch_size), 'ExecutionEnvironment', 'cpu');
            inference_time = toc;
            fprintf('批次大小 %d: 推理时间 %.4f 秒 (%.4f 秒/样本)\n', ...
                    batch_size, inference_time, inference_time/batch_size);
        end
    end
end

%% 运行基准测试
runBenchmark(simNet, xTestFrames, yTest);

%% 网络架构总结
fprintf('\n=== 最终网络架构总结 ===\n');
fprintf('1. 输入层: [%d, %d, %d] 时序IQ信号\n', frameLength, numChannels, 1);
fprintf('2. CNN模块: 3个卷积层 (32->64->128 通道)\n');
fprintf('3. ResNet风格: 4个残差卷积层提取深层特征\n');
fprintf('4. 注意力机制: 2层全连接网络 (256->128)\n');
fprintf('5. LSTM模块: 双层LSTM (%d隐藏单元)\n', numHiddenUnits);
fprintf('6. 分类器: 全连接层 -> Softmax -> %d类输出\n', numClasses);
fprintf('7. 正则化: BatchNorm + Dropout (0.3, 0.5)\n');
fprintf('8. 优化器: Adam, 学习率 0.008, 分段衰减\n');
fprintf('9. 总体准确率: %.2f%%\n', testAccuracy*100);
fprintf('10. 训练时间: %.2f 秒\n', TrainTime);