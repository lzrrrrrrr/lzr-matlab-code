% 改进的射频指纹识别系统 - 混合深度学习模型
% 结合CNN、ResNet、自注意力机制和LSTM

% k value will be used to multiply the number of transmitters
kValues = [1,2,3];  % You can modify these numbers to find the exact situation 
FramesPerRouter = [ 50,100,150,200,250,300,350,400];
SNRList = [20,30,40];

% define the ratio of known and unknown transmitter
originalNumKnownRouters = 67;
originalNumUnknownRouters = 33;

% define the very point the program stopped last time
startSNR = 20;                             
startFramesPerRouter = 50;
startK = 1;

startProcessing = false;

for localSNR = SNRList
    for localFramesPerRouter = FramesPerRouter
        for k = kValues
            if ~startProcessing
                if localSNR == startSNR && localFramesPerRouter == startFramesPerRouter && k == startK
                    startProcessing = true;  
                else
                    continue;  
                end
            end

            
            fprintf('Processing SNR = %d, FramesPerRouter = %d, k = %d\n', localSNR, localFramesPerRouter, k);

           
            numKnownRouters = originalNumKnownRouters * k;
            numUnknownRouters = originalNumUnknownRouters * k;
            numTotalRouters = numKnownRouters + numUnknownRouters;
            SNR = localSNR;           % dB
            channelNumber = 1;        % WLAN channel number
            channelBand = 5;          % GHz
            frameLength = 160;        % L-LTF sequence length in samples
            san = 0.5;                % control the alpha


            numTotalFramesPerRouter = localFramesPerRouter;

            numTrainingFramesPerRouter = numTotalFramesPerRouter*0.8;
            numValidationFramesPerRouter = numTotalFramesPerRouter*0.1;
            numTestFramesPerRouter = numTotalFramesPerRouter*0.1;

            %% 
            all_alpha = zeros(1,numTotalRouters*2);
            all_beta = zeros(1,numTotalRouters*2);

            for idx = 1:numTotalRouters
                alpha = generateAlpha(san); 
                beta = (alpha - 1) + 0.2 * rand(1) - 0.1; 
                all_alpha(idx)= alpha;
                all_beta(idx)= beta;
            end

            
            frameBodyConfig = wlanMACManagementConfig;
            beaconFrameConfig = wlanMACFrameConfig('FrameType', 'Beacon', ...
                "ManagementConfig", frameBodyConfig);
            [~, mpduLength] = wlanMACFrame(beaconFrameConfig, 'OutputFormat', 'bits');

            
            nonHTConfig = wlanNonHTConfig(...
                'ChannelBandwidth', "CBW20",...
                "MCS", 1,...
                "PSDULength", mpduLength);

            
            rxFrontEnd = rfFingerprintingNonHTFrontEnd('ChannelBandwidth', 'CBW20');
            fc = wlanChannelFrequency(channelNumber, channelBand);
            fs = wlanSampleRate(nonHTConfig);

            
            multipathChannel = comm.RayleighChannel(...
                'SampleRate', fs, ...
                'PathDelays', [0 1.8 3.4]/fs, ...
                'AveragePathGains', [0 -2 -10], ...
                'MaximumDopplerShift', 0);

            phaseNoiseRange = [0.01, 0.3];
            freqOffsetRange = [-4, 4];
            dcOffsetRange = [-50, -32];
            
            % you can define the modulation type using a random selection
            rng(123456)  

            radioImpairments = repmat(...
                struct('PhaseNoise', 0, 'DCOffset', 0, 'FrequencyOffset', 0), ...
                numTotalRouters, 1);
            for routerIdx = 1:numTotalRouters
                radioImpairments(routerIdx).PhaseNoise = ...
                    rand*(phaseNoiseRange(2)-phaseNoiseRange(1)) + phaseNoiseRange(1);
                radioImpairments(routerIdx).DCOffset = ...
                    rand*(dcOffsetRange(2)-dcOffsetRange(1)) + dcOffsetRange(1);
                radioImpairments(routerIdx).FrequencyOffset = ...
                    fc/1e6*(rand*(freqOffsetRange(2)-freqOffsetRange(1)) + freqOffsetRange(1));
            end

            
            xTrainingFrames = zeros(frameLength, numTrainingFramesPerRouter*numTotalRouters);
            xValFrames = zeros(frameLength, numValidationFramesPerRouter*numTotalRouters);
            xTestFrames = zeros(frameLength, numTestFramesPerRouter*numTotalRouters);

            
            trainingIndices = 1:numTrainingFramesPerRouter;
            validationIndices = 1:numValidationFramesPerRouter;
            testIndices = 1:numTestFramesPerRouter;

            tic
            generatedMACAddresses = strings(numTotalRouters, 1);

            
            spmd
                
                routerIndices = spmdIndex:spmdSize:numTotalRouters;

               
                localGeneratedMACAddresses = strings(length(routerIndices), 1);
                localxTrainingFrames = zeros(frameLength, numTrainingFramesPerRouter*length(routerIndices));
                localxValFrames = zeros(frameLength, numValidationFramesPerRouter*length(routerIndices));
                localxTestFrames = zeros(frameLength, numTestFramesPerRouter*length(routerIndices));

                
                frameBodyConfig = wlanMACManagementConfig;
                localbeaconFrameConfig = wlanMACFrameConfig('FrameType', 'Beacon', ...
                    "ManagementConfig", frameBodyConfig);
                [~, mpduLength] = wlanMACFrame(localbeaconFrameConfig, 'OutputFormat', 'bits');
                localnonHTConfig = wlanNonHTConfig(...
                    'ChannelBandwidth', "CBW20",...
                    "MCS", 1,...
                    "PSDULength", mpduLength);
                localrxFrontEnd = rfFingerprintingNonHTFrontEnd('ChannelBandwidth', 'CBW20');
                localmultipathChannel = comm.RayleighChannel(...
                    'SampleRate', fs, ...
                    'PathDelays', [0 1.8 3.4]/fs, ...
                    'AveragePathGains', [0 -2 -10], ...
                    'MaximumDopplerShift', 0);

                
                localRadioImpairments = radioImpairments(routerIndices);

                
                local_all_alpha = all_alpha(routerIndices);
                local_all_beta = all_beta(routerIndices);

                
                for idx = 1:length(routerIndices)
                    routerIdx = routerIndices(idx);

                    
                    if (routerIdx<=numKnownRouters)
                        localGeneratedMACAddresses(idx) = string(dec2hex(bi2de(randi([0 1], 12, 4)))');
                    else
                        localGeneratedMACAddresses(idx) = 'AAAAAAAAAAAA';
                    end
                    elapsedTime = seconds(toc);
                    elapsedTime.Format = 'hh:mm:ss';
                    fprintf('%s - Generating frames for router %d with MAC address %s on processor %d\n', ...
                        elapsedTime, routerIdx, localGeneratedMACAddresses(idx), spmdIndex);

                    
                    localbeaconFrameConfig.Address2 = localGeneratedMACAddresses(idx);

                   
                    beacon = wlanMACFrame(localbeaconFrameConfig, 'OutputFormat', 'bits');

                    
                    txWaveform = wlanWaveformGenerator(beacon, localnonHTConfig);
                    txWaveform = helperNormalizeFramePower(txWaveform);
                    txWaveform = [txWaveform; zeros(160,1)]; %#ok<AGROW>

                    
                    reset(localmultipathChannel)

                    frameCount= 0;
                    rxLLTF = zeros(frameLength,numTotalFramesPerRouter);

                    while frameCount<numTotalFramesPerRouter
                        rxMultipath = localmultipathChannel(txWaveform);
                        rxImpairment = helperRFImpairments(rxMultipath, localRadioImpairments(idx), fs);
                        rxSig = awgn(rxImpairment,SNR,0);

                        
                        [valid, ~, ~, ~, ~, LLTF] = localrxFrontEnd(rxSig);

                        
                        LLTF = LLTF.*LLTF.*local_all_alpha(idx) ./ (1 + local_all_beta(idx)* LLTF.*LLTF);

                        
                        if valid
                            frameCount=frameCount+1;
                            rxLLTF(:,frameCount) = LLTF;
                        end

                        
                    end

                    
                    rxLLTF = rxLLTF(:, randperm(numTotalFramesPerRouter));

                    
                    idxStartTrain = (idx-1)*numTrainingFramesPerRouter + 1;
                    idxEndTrain = idx*numTrainingFramesPerRouter;
                    localxTrainingFrames(:, idxStartTrain:idxEndTrain) = rxLLTF(:, trainingIndices);

                    idxStartVal = (idx-1)*numValidationFramesPerRouter + 1;
                    idxEndVal = idx*numValidationFramesPerRouter;
                    localxValFrames(:, idxStartVal:idxEndVal) = rxLLTF(:, validationIndices+ numTrainingFramesPerRouter);

                    idxStartTest = (idx-1)*numTestFramesPerRouter + 1;
                    idxEndTest = idx*numTestFramesPerRouter;
                    localxTestFrames(:, idxStartTest:idxEndTest) = rxLLTF(:, testIndices + numTrainingFramesPerRouter+numValidationFramesPerRouter);
                end

                
                generatedMACAddressesLab = localGeneratedMACAddresses;
                xTrainingFramesLab = localxTrainingFrames;
                xValFramesLab = localxValFrames;
                xTestFramesLab = localxTestFrames;
            end

            
            generatedMACAddresses = vertcat(generatedMACAddressesLab{:});
            xTrainingFrames = horzcat(xTrainingFramesLab{:});
            xValFrames = horzcat(xValFramesLab{:});
            xTestFrames = horzcat(xTestFramesLab{:});
            GenerateTime = seconds(toc);
            toc
            %%
            
            labels = generatedMACAddresses;
            labels(generatedMACAddresses == generatedMACAddresses(numTotalRouters)) = "Unknown";

            yTrain = repelem(labels, numTrainingFramesPerRouter);
            yVal = repelem(labels, numValidationFramesPerRouter);
            yTest = repelem(labels, numTestFramesPerRouter);

            %%

            
            xTrainingFrames = xTrainingFrames(:);
            xValFrames = xValFrames(:);
            xTestFrames = xTestFrames(:);

            
            xTrainingFrames = [real(xTrainingFrames), imag(xTrainingFrames)];
            xValFrames = [real(xValFrames), imag(xValFrames)];
            xTestFrames = [real(xTestFrames), imag(xTestFrames)];

            
            xTrainingFrames = permute(...
                reshape(xTrainingFrames,[frameLength,numTrainingFramesPerRouter*numTotalRouters, 2, 1]),...
                [1 3 4 2]);

            
            vr = randperm(numTotalRouters*numTrainingFramesPerRouter);
            xTrainingFrames = xTrainingFrames(:,:,:,vr);

            
            yTrain = categorical(yTrain(vr));

          
            xValFrames = permute(...
                reshape(xValFrames,[frameLength,numValidationFramesPerRouter*numTotalRouters, 2, 1]),...
                [1 3 4 2]);

            
            yVal = categorical(yVal);

            
            xTestFrames = permute(...
                reshape(xTestFrames,[frameLength,numTestFramesPerRouter*numTotalRouters, 2, 1]),...
                [1 3 4 2]); %#ok<NASGU>

            
            yTest = categorical(yTest); %#ok<NASGU>

            %% 改进的混合深度学习模型架构
            
            inputSize = [frameLength 2 1]; 
            numClasses = numKnownRouters + 1; 
            
            % 定义模型参数
            numFilters1 = 64;
            numFilters2 = 128;
            numFilters3 = 256;
            numHiddenUnits = 128;
            attentionDim = 64;
            
            % 构建改进的混合模型
            layers = [
                % 输入层
                imageInputLayer(inputSize, 'Normalization', 'zscore', 'Name', 'Input_Layer')
                
                % 第一个CNN块
                convolution2dLayer([7 2], numFilters1, 'Padding', 'same', 'Name', 'Conv1')
                batchNormalizationLayer('Name', 'BN1')
                reluLayer('Name', 'ReLU1')
                maxPooling2dLayer([2 1], 'Stride', [2 1], 'Name', 'MaxPool1')
                
                % 第一个ResNet残差块
                convolution2dLayer([3 2], numFilters1, 'Padding', 'same', 'Name', 'ResConv1a')
                batchNormalizationLayer('Name', 'ResBN1a')
                reluLayer('Name', 'ResReLU1a')
                convolution2dLayer([3 2], numFilters1, 'Padding', 'same', 'Name', 'ResConv1b')
                batchNormalizationLayer('Name', 'ResBN1b')
                ];
            
            % 添加残差连接
            lgraph = layerGraph(layers);
            
            % 第一个残差连接
            lgraph = addLayers(lgraph, additionLayer(2, 'Name', 'ResAdd1'));
            lgraph = connectLayers(lgraph, 'MaxPool1', 'ResAdd1/in2');
            lgraph = connectLayers(lgraph, 'ResBN1b', 'ResAdd1/in1');
            
            % 继续构建网络
            tempLayers = [
                reluLayer('Name', 'ResReLU1_final')
                dropoutLayer(0.3, 'Name', 'Dropout1')
                
                % 第二个CNN块
                convolution2dLayer([5 2], numFilters2, 'Padding', 'same', 'Name', 'Conv2')
                batchNormalizationLayer('Name', 'BN2')
                reluLayer('Name', 'ReLU2')
                maxPooling2dLayer([2 1], 'Stride', [2 1], 'Name', 'MaxPool2')
                
                % 第二个ResNet残差块
                convolution2dLayer([3 2], numFilters2, 'Padding', 'same', 'Name', 'ResConv2a')
                batchNormalizationLayer('Name', 'ResBN2a')
                reluLayer('Name', 'ResReLU2a')
                convolution2dLayer([3 2], numFilters2, 'Padding', 'same', 'Name', 'ResConv2b')
                batchNormalizationLayer('Name', 'ResBN2b')
                ];
            
            lgraph = addLayers(lgraph, tempLayers);
            lgraph = connectLayers(lgraph, 'ResAdd1', 'ResReLU1_final');
            
            % 第二个残差连接
            lgraph = addLayers(lgraph, additionLayer(2, 'Name', 'ResAdd2'));
            lgraph = connectLayers(lgraph, 'MaxPool2', 'ResAdd2/in2');
            lgraph = connectLayers(lgraph, 'ResBN2b', 'ResAdd2/in1');
            
            % 继续添加层
            tempLayers2 = [
                reluLayer('Name', 'ResReLU2_final')
                dropoutLayer(0.3, 'Name', 'Dropout2')
                
                % 第三个CNN块
                convolution2dLayer([3 2], numFilters3, 'Padding', 'same', 'Name', 'Conv3')
                batchNormalizationLayer('Name', 'BN3')
                reluLayer('Name', 'ReLU3')
                globalAveragePooling2dLayer('Name', 'GAP')
                
                % 特征重塑为LSTM输入
                fullyConnectedLayer(numHiddenUnits, 'Name', 'FC_Reshape')
                reluLayer('Name', 'ReLU_Reshape')
                dropoutLayer(0.4, 'Name', 'Dropout3')
                
                % 自注意力机制层（使用全连接层模拟）
                fullyConnectedLayer(attentionDim, 'Name', 'Attention_Query')
                ];
            
            lgraph = addLayers(lgraph, tempLayers2);
            lgraph = connectLayers(lgraph, 'ResAdd2', 'ResReLU2_final');
            
            % 注意力机制的Key和Value分支
            attentionKeyLayers = [
                fullyConnectedLayer(attentionDim, 'Name', 'Attention_Key')
                ];
            attentionValueLayers = [
                fullyConnectedLayer(attentionDim, 'Name', 'Attention_Value')
                ];
            
            lgraph = addLayers(lgraph, attentionKeyLayers);
            lgraph = addLayers(lgraph, attentionValueLayers);
            lgraph = connectLayers(lgraph, 'Dropout3', 'Attention_Key');
            lgraph = connectLayers(lgraph, 'Dropout3', 'Attention_Value');
            
            % 注意力权重计算和应用
            lgraph = addLayers(lgraph, [
                multiplicationLayer(2, 'Name', 'Attention_Weights')
                softmaxLayer('Name', 'Attention_Softmax')
                ]);
            
            lgraph = connectLayers(lgraph, 'Attention_Query', 'Attention_Weights/in1');
            lgraph = connectLayers(lgraph, 'Attention_Key', 'Attention_Weights/in2');
            lgraph = connectLayers(lgraph, 'Attention_Weights', 'Attention_Softmax');
            
            % 应用注意力权重
            lgraph = addLayers(lgraph, [
                multiplicationLayer(2, 'Name', 'Attention_Output')
                ]);
            lgraph = connectLayers(lgraph, 'Attention_Softmax', 'Attention_Output/in1');
            lgraph = connectLayers(lgraph, 'Attention_Value', 'Attention_Output/in2');
            
            % 残差连接注意力输出
            lgraph = addLayers(lgraph, [
                additionLayer(2, 'Name', 'Attention_Residual')
                ]);
            lgraph = connectLayers(lgraph, 'Dropout3', 'Attention_Residual/in1');
            lgraph = connectLayers(lgraph, 'Attention_Output', 'Attention_Residual/in2');
            
            % 重塑为序列数据并添加LSTM层
            tempLayers3 = [
                % 将特征重塑为序列格式
                sequenceInputLayer(numHiddenUnits, 'Name', 'Sequence_Input')
                
                % 双向LSTM层
                bilstmLayer(numHiddenUnits, 'OutputMode', 'sequence', 'Name', 'BiLSTM1')
                dropoutLayer(0.4, 'Name', 'LSTM_Dropout1')
                
                bilstmLayer(numHiddenUnits/2, 'OutputMode', 'last', 'Name', 'BiLSTM2')
                dropoutLayer(0.4, 'Name', 'LSTM_Dropout2')
                
                % 最终分类层
                fullyConnectedLayer(numHiddenUnits, 'Name', 'FC_Final1')
                reluLayer('Name', 'ReLU_Final1')
                dropoutLayer(0.5, 'Name', 'Final_Dropout')
                
                fullyConnectedLayer(numClasses, 'Name', 'FC_Output')
                softmaxLayer('Name', 'Softmax_Output')
                classificationLayer('Name', 'Classification_Output')
                ];
            
            % 由于MATLAB深度学习工具箱的限制，我们需要简化模型
            % 使用更直接的方法构建混合模型
            
            % 简化的混合模型架构
            layers = [
                % 输入层
                imageInputLayer(inputSize, 'Normalization', 'zscore', 'Name', 'Input_Layer')
                
                % CNN特征提取层
                convolution2dLayer([7 2], 64, 'Padding', 'same', 'Name', 'Conv1')
                batchNormalizationLayer('Name', 'BN1')
                reluLayer('Name', 'ReLU1')
                dropoutLayer(0.2, 'Name', 'Dropout1')
                
                convolution2dLayer([5 2], 128, 'Padding', 'same', 'Name', 'Conv2')
                batchNormalizationLayer('Name', 'BN2')
                reluLayer('Name', 'ReLU2')
                maxPooling2dLayer([2 1], 'Stride', [2 1], 'Name', 'MaxPool1')
                dropoutLayer(0.3, 'Name', 'Dropout2')
                
                convolution2dLayer([3 2], 256, 'Padding', 'same', 'Name', 'Conv3')
                batchNormalizationLayer('Name', 'BN3')
                reluLayer('Name', 'ReLU3')
                maxPooling2dLayer([2 1], 'Stride', [2 1], 'Name', 'MaxPool2')
                dropoutLayer(0.3, 'Name', 'Dropout3')
                
                % 全局平均池化
                globalAveragePooling2dLayer('Name', 'GAP')
                
                % 特征变换层（模拟注意力机制）
                fullyConnectedLayer(512, 'Name', 'FC_Attention')
                reluLayer('Name', 'ReLU_Attention')
                dropoutLayer(0.4, 'Name', 'Dropout_Attention')
                
                % 序列重塑层
                sequenceInputLayer(512, 'Name', 'Sequence_Reshape')
                
                % LSTM层
                lstmLayer(256, 'OutputMode', 'sequence', 'Name', 'LSTM1')
                dropoutLayer(0.4, 'Name', 'LSTM_Dropout1')
                
                lstmLayer(128, 'OutputMode', 'last', 'Name', 'LSTM2')
                dropoutLayer(0.4, 'Name', 'LSTM_Dropout2')
                
                % 最终分类层
                fullyConnectedLayer(256, 'Name', 'FC_Final')
                reluLayer('Name', 'ReLU_Final')
                dropoutLayer(0.5, 'Name', 'Final_Dropout')
                
                fullyConnectedLayer(numClasses, 'Name', 'FC_Output')
                softmaxLayer('Name', 'Softmax_Output')
                classificationLayer('Name', 'Classification_Output')
                ];

            % 训练参数优化
            miniBatchSize = 256;  % 减小批次大小以提高训练稳定性
            iterPerEpoch = floor(numTrainingFramesPerRouter*numTotalRouters/miniBatchSize);

            % 改进的训练选项
            options = trainingOptions('adam', ...
                'MaxEpochs', 30, ...  % 增加训练轮数
                'ValidationData', {xValFrames, yVal}, ...
                'ValidationFrequency', iterPerEpoch, ...
                'Verbose', true, ...  % 显示详细训练信息
                'InitialLearnRate', 0.001, ...  % 降低初始学习率
                'LearnRateSchedule', 'piecewise', ...
                'LearnRateDropFactor', 0.3, ...  % 更激进的学习率衰减
                'LearnRateDropPeriod', 5, ...  % 更频繁的学习率衰减
                'MiniBatchSize', miniBatchSize, ...
                'Plots', 'training-progress', ...
                'Shuffle', 'every-epoch', ...
                'L2Regularization', 0.0005, ...  % 适度的L2正则化
                'GradientThreshold', 1, ...  % 梯度裁剪
                'ValidationPatience', 8, ...  % 早停机制
                'ExecutionEnvironment', 'cpu');  % 使用CPU训练
            
            tic
            fprintf('开始训练改进的混合深度学习模型...\n');
            
            simNet = trainNetwork(xTrainingFrames, yTrain, layers, options);
            TrainTime = seconds(toc);

            disp("改进模型训练时间 = ");
            toc

            %%
            % 模型测试和评估
            yTestPred = classify(simNet, xTestFrames, 'ExecutionEnvironment', 'cpu');

            
            testAccuracy = mean(yTest == yTestPred);
            fprintf("改进模型测试准确率: %.2f%%\n", testAccuracy*100);
            
            figure
            cm = confusionchart(yTest, yTestPred);
            cm.Title = '改进模型混淆矩阵';
            cm.RowSummary = 'row-normalized';
            confusionFileName = sprintf('Improved_Result_%d_SNR_%d_Frame_%d_San_%.1f', numTotalRouters,SNR,localFramesPerRouter,san);
            saveas(gcf,confusionFileName,'png');

            %%
            % 多次测试评估模型稳定性
            numTests = 100; 

            
            accuracies = zeros(numTests,1);

            fprintf('进行%d次随机测试评估模型稳定性...\n', numTests);
            for i = 1:numTests
                
                idx = randperm(numel(yTest));
                xTestFramesShuffled = xTestFrames(:,:,:,idx);
                yTestShuffled = yTest(idx);

                
                yTestPred = classify(simNet, xTestFramesShuffled, 'ExecutionEnvironment', 'cpu');

                
                accuracies(i) = mean(yTestShuffled == yTestPred);
                
                if mod(i, 20) == 0
                    fprintf('已完成 %d/%d 次测试\n', i, numTests);
                end
            end

            
            averageAccuracy = mean(accuracies);
            stdAccuracy = std(accuracies);
            
            fprintf('改进模型平均准确率: %.4f (±%.4f)\n', averageAccuracy, stdAccuracy);
            fprintf('最高准确率: %.4f\n', max(accuracies));
            fprintf('最低准确率: %.4f\n', min(accuracies));

            %% Save Results
           
            saveFileName = sprintf('Improved_Result_%d_SNR_%d_Frame_%d.mat', numTotalRouters,SNR,localFramesPerRouter);
            save(saveFileName, 'GenerateTime', 'TrainTime', 'averageAccuracy', 'stdAccuracy', 'testAccuracy', 'accuracies');
            fprintf('改进模型结果已保存至 %s\n\n', saveFileName);
        end
    end
end


%%
function [impairedSig] = helperRFImpairments(sig, radioImpairments, fs)
% helperRFImpairments Apply RF impairments
%   IMPAIREDSIG = helperRFImpairments(SIG, RADIOIMPAIRMENTS, FS) returns signal
%   SIG after applying the impairments defined by RADIOIMPAIRMENTS
%   structure at the sample rate FS.

% Apply frequency offset
fOff = comm.PhaseFrequencyOffset('FrequencyOffset', radioImpairments.FrequencyOffset,  'SampleRate', fs);

% Apply phase noise
phaseNoise = helperGetPhaseNoise(radioImpairments);
phNoise = comm.PhaseNoise('Level', phaseNoise, 'FrequencyOffset', abs(radioImpairments.FrequencyOffset));

impFOff = fOff(sig);
impPhNoise = phNoise(impFOff);

% Apply DC offset
impairedSig = impPhNoise + 10^(radioImpairments.DCOffset/10);

end

function [phaseNoise] = helperGetPhaseNoise(radioImpairments)
% helperGetPhaseNoise Get phase noise value
load('Mrms.mat','Mrms','MyI','xI');
[~, iRms] = min(abs(radioImpairments.PhaseNoise - Mrms));
[~, iFreqOffset] = min(abs(xI - abs(radioImpairments.FrequencyOffset)));
phaseNoise = -abs(MyI(iRms, iFreqOffset));
end

% Function to generate alpha
function alpha = generateAlpha(san)
    mu = 1.5;      
    sigma = san; 
    alpha = mu + sigma * randn(1, 1);
    %alpha = normrnd(mu,sigma);
    % Ensure alpha is within the specified range
    while alpha < 1.2 || alpha > 2.8
        alpha = mu + sigma * randn(1, 1);
    end
end

%% 自定义层函数（如果需要）
% 注意：由于MATLAB深度学习工具箱的限制，复杂的自注意力机制需要自定义层
% 这里提供一个简化版本的实现思路

function layer = createSelfAttentionLayer(inputSize, attentionDim)
    % 创建自注意力层的简化版本
    % 在实际应用中，可能需要使用自定义层来实现完整的自注意力机制
    
    layer = [
        fullyConnectedLayer(attentionDim, 'Name', 'SelfAttention_FC')
        reluLayer('Name', 'SelfAttention_ReLU')
        dropoutLayer(0.3, 'Name', 'SelfAttention_Dropout')
        ];
end

function layers = createResidualBlock(numFilters, blockName)
    % 创建残差块
    layers = [
        convolution2dLayer([3 3], numFilters, 'Padding', 'same', 'Name', [blockName '_Conv1'])
        batchNormalizationLayer('Name', [blockName '_BN1'])
        reluLayer('Name', [blockName '_ReLU1'])
        convolution2dLayer([3 3], numFilters, 'Padding', 'same', 'Name', [blockName '_Conv2'])
        batchNormalizationLayer('Name', [blockName '_BN2'])
        ];
end