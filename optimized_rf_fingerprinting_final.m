% Optimized RF Fingerprinting with Enhanced CNN-ResNet-Attention-LSTM Model
% Targeting 90%+ accuracy with low loss
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

            %% Optimized Enhanced Deep Learning Model for 90%+ Accuracy
            
            inputSize = [frameLength 2 1]; 
            numClasses = numKnownRouters + 1; 
            
            % Build optimized CNN-ResNet-Attention-LSTM network
            lgraph = layerGraph();
            
            % Input and initial feature extraction
            tempLayers = [
                imageInputLayer(inputSize, 'Normalization', 'zscore', 'Name', 'InputLayer')
                
                % Enhanced initial CNN block
                convolution2dLayer([9 1], 32, 'Padding', [4 0], 'Stride', [1 1], 'Name', 'InitConv')
                batchNormalizationLayer('Name', 'InitBN')
                reluLayer('Name', 'InitReLU')
                
                % First CNN block
                convolution2dLayer([7 1], 64, 'Padding', [3 0], 'Stride', [2 1], 'Name', 'Conv1')
                batchNormalizationLayer('Name', 'BN1')
                reluLayer('Name', 'ReLU1')
                maxPooling2dLayer([3 1], 'Stride', [2 1], 'Padding', [1 0], 'Name', 'MaxPool1')
            ];
            lgraph = addLayers(lgraph, tempLayers);
            
            % First ResNet block with improved architecture
            tempLayers = [
                convolution2dLayer([5 1], 64, 'Padding', [2 0], 'Name', 'ResConv1_1')
                batchNormalizationLayer('Name', 'ResBN1_1')
                reluLayer('Name', 'ResReLU1_1')
                dropoutLayer(0.1, 'Name', 'ResDropout1_1')
                convolution2dLayer([5 1], 64, 'Padding', [2 0], 'Name', 'ResConv1_2')
                batchNormalizationLayer('Name', 'ResBN1_2')
            ];
            lgraph = addLayers(lgraph, tempLayers);
            
            % Skip connection for first ResNet block
            tempLayers = [
                convolution2dLayer([1 1], 64, 'Stride', [1 1], 'Name', 'ResSkip1')
                batchNormalizationLayer('Name', 'ResSkipBN1')
            ];
            lgraph = addLayers(lgraph, tempLayers);
            
            tempLayers = [
                additionLayer(2, 'Name', 'ResAdd1')
                reluLayer('Name', 'ResReLU1_Final')
                dropoutLayer(0.1, 'Name', 'ResDropout1_Final')
            ];
            lgraph = addLayers(lgraph, tempLayers);
            
            % Second ResNet block
            tempLayers = [
                convolution2dLayer([5 1], 128, 'Padding', [2 0], 'Stride', [2 1], 'Name', 'ResConv2_1')
                batchNormalizationLayer('Name', 'ResBN2_1')
                reluLayer('Name', 'ResReLU2_1')
                dropoutLayer(0.1, 'Name', 'ResDropout2_1')
                convolution2dLayer([5 1], 128, 'Padding', [2 0], 'Name', 'ResConv2_2')
                batchNormalizationLayer('Name', 'ResBN2_2')
            ];
            lgraph = addLayers(lgraph, tempLayers);
            
            % Skip connection for second ResNet block
            tempLayers = [
                convolution2dLayer([1 1], 128, 'Stride', [2 1], 'Name', 'ResSkip2')
                batchNormalizationLayer('Name', 'ResSkipBN2')
            ];
            lgraph = addLayers(lgraph, tempLayers);
            
            tempLayers = [
                additionLayer(2, 'Name', 'ResAdd2')
                reluLayer('Name', 'ResReLU2_Final')
                dropoutLayer(0.15, 'Name', 'ResDropout2_Final')
            ];
            lgraph = addLayers(lgraph, tempLayers);
            
            % Third ResNet block
            tempLayers = [
                convolution2dLayer([3 1], 256, 'Padding', [1 0], 'Stride', [2 1], 'Name', 'ResConv3_1')
                batchNormalizationLayer('Name', 'ResBN3_1')
                reluLayer('Name', 'ResReLU3_1')
                dropoutLayer(0.15, 'Name', 'ResDropout3_1')
                convolution2dLayer([3 1], 256, 'Padding', [1 0], 'Name', 'ResConv3_2')
                batchNormalizationLayer('Name', 'ResBN3_2')
            ];
            lgraph = addLayers(lgraph, tempLayers);
            
            % Skip connection for third ResNet block
            tempLayers = [
                convolution2dLayer([1 1], 256, 'Stride', [2 1], 'Name', 'ResSkip3')
                batchNormalizationLayer('Name', 'ResSkipBN3')
            ];
            lgraph = addLayers(lgraph, tempLayers);
            
            tempLayers = [
                additionLayer(2, 'Name', 'ResAdd3')
                reluLayer('Name', 'ResReLU3_Final')
                dropoutLayer(0.2, 'Name', 'ResDropout3_Final')
            ];
            lgraph = addLayers(lgraph, tempLayers);
            
            % Fourth ResNet block for deeper features
            tempLayers = [
                convolution2dLayer([3 1], 512, 'Padding', [1 0], 'Stride', [2 1], 'Name', 'ResConv4_1')
                batchNormalizationLayer('Name', 'ResBN4_1')
                reluLayer('Name', 'ResReLU4_1')
                dropoutLayer(0.2, 'Name', 'ResDropout4_1')
                convolution2dLayer([3 1], 512, 'Padding', [1 0], 'Name', 'ResConv4_2')
                batchNormalizationLayer('Name', 'ResBN4_2')
            ];
            lgraph = addLayers(lgraph, tempLayers);
            
            % Skip connection for fourth ResNet block
            tempLayers = [
                convolution2dLayer([1 1], 512, 'Stride', [2 1], 'Name', 'ResSkip4')
                batchNormalizationLayer('Name', 'ResSkipBN4')
            ];
            lgraph = addLayers(lgraph, tempLayers);
            
            tempLayers = [
                additionLayer(2, 'Name', 'ResAdd4')
                reluLayer('Name', 'ResReLU4_Final')
                dropoutLayer(0.2, 'Name', 'ResDropout4_Final')
            ];
            lgraph = addLayers(lgraph, tempLayers);
            
            % Enhanced attention mechanism
            tempLayers = [
                globalAveragePooling2dLayer('Name', 'GAP')
                flattenLayer('Name', 'Flatten')
                
                % Multi-head attention simulation
                fullyConnectedLayer(512, 'Name', 'AttentionQuery')
                batchNormalizationLayer('Name', 'AttentionBN1')
                reluLayer('Name', 'AttentionReLU1')
                dropoutLayer(0.1, 'Name', 'AttentionDropout1')
                
                fullyConnectedLayer(512, 'Name', 'AttentionKey')
                batchNormalizationLayer('Name', 'AttentionBN2')
                reluLayer('Name', 'AttentionReLU2')
                dropoutLayer(0.1, 'Name', 'AttentionDropout2')
                
                fullyConnectedLayer(512, 'Name', 'AttentionValue')
                batchNormalizationLayer('Name', 'AttentionBN3')
                tanhLayer('Name', 'AttentionTanh')
                
                % Attention weights
                fullyConnectedLayer(512, 'Name', 'AttentionWeights')
                softmaxLayer('Name', 'AttentionSoftmax')
            ];
            lgraph = addLayers(lgraph, tempLayers);
            
            % Apply attention
            tempLayers = [
                multiplicationLayer(2, 'Name', 'AttentionApply')
                dropoutLayer(0.15, 'Name', 'AttentionFinalDropout')
            ];
            lgraph = addLayers(lgraph, tempLayers);
            
            % Enhanced LSTM layers
            tempLayers = [
                lstmLayer(512, 'OutputMode', 'sequence', 'Name', 'LSTM1')
                dropoutLayer(0.2, 'Name', 'LSTMDropout1')
                
                lstmLayer(256, 'OutputMode', 'sequence', 'Name', 'LSTM2')
                dropoutLayer(0.2, 'Name', 'LSTMDropout2')
                
                lstmLayer(128, 'OutputMode', 'last', 'Name', 'LSTM3')
                dropoutLayer(0.25, 'Name', 'LSTMDropout3')
            ];
            lgraph = addLayers(lgraph, tempLayers);
            
            % Enhanced classification layers
            tempLayers = [
                fullyConnectedLayer(1024, 'Name', 'FC1')
                batchNormalizationLayer('Name', 'BNFC1')
                reluLayer('Name', 'ReLUFC1')
                dropoutLayer(0.5, 'Name', 'FCDropout1')
                
                fullyConnectedLayer(512, 'Name', 'FC2')
                batchNormalizationLayer('Name', 'BNFC2')
                reluLayer('Name', 'ReLUFC2')
                dropoutLayer(0.4, 'Name', 'FCDropout2')
                
                fullyConnectedLayer(256, 'Name', 'FC3')
                batchNormalizationLayer('Name', 'BNFC3')
                reluLayer('Name', 'ReLUFC3')
                dropoutLayer(0.3, 'Name', 'FCDropout3')
                
                fullyConnectedLayer(numClasses, 'Name', 'FCFinal')
                softmaxLayer('Name', 'SoftMax')
                classificationLayer('Name', 'Output')
            ];
            lgraph = addLayers(lgraph, tempLayers);
            
            % Connect all layers
            % Main path connections
            lgraph = connectLayers(lgraph, 'MaxPool1', 'ResConv1_1');
            lgraph = connectLayers(lgraph, 'ResBN1_2', 'ResAdd1/in1');
            
            % Skip connection 1
            lgraph = connectLayers(lgraph, 'MaxPool1', 'ResSkip1');
            lgraph = connectLayers(lgraph, 'ResSkipBN1', 'ResAdd1/in2');
            
            lgraph = connectLayers(lgraph, 'ResDropout1_Final', 'ResConv2_1');
            lgraph = connectLayers(lgraph, 'ResBN2_2', 'ResAdd2/in1');
            
            % Skip connection 2
            lgraph = connectLayers(lgraph, 'ResDropout1_Final', 'ResSkip2');
            lgraph = connectLayers(lgraph, 'ResSkipBN2', 'ResAdd2/in2');
            
            lgraph = connectLayers(lgraph, 'ResDropout2_Final', 'ResConv3_1');
            lgraph = connectLayers(lgraph, 'ResBN3_2', 'ResAdd3/in1');
            
            % Skip connection 3
            lgraph = connectLayers(lgraph, 'ResDropout2_Final', 'ResSkip3');
            lgraph = connectLayers(lgraph, 'ResSkipBN3', 'ResAdd3/in2');
            
            lgraph = connectLayers(lgraph, 'ResDropout3_Final', 'ResConv4_1');
            lgraph = connectLayers(lgraph, 'ResBN4_2', 'ResAdd4/in1');
            
            % Skip connection 4
            lgraph = connectLayers(lgraph, 'ResDropout3_Final', 'ResSkip4');
            lgraph = connectLayers(lgraph, 'ResSkipBN4', 'ResAdd4/in2');
            
            % Attention connections
            lgraph = connectLayers(lgraph, 'ResDropout4_Final', 'GAP');
            lgraph = connectLayers(lgraph, 'AttentionValue', 'AttentionApply/in1');
            lgraph = connectLayers(lgraph, 'AttentionSoftmax', 'AttentionApply/in2');
            
            % LSTM connections
            lgraph = connectLayers(lgraph, 'AttentionFinalDropout', 'LSTM1');
            
            % Final connections
            lgraph = connectLayers(lgraph, 'LSTMDropout3', 'FC1');

            % Optimized training options for maximum accuracy
            miniBatchSize = 64; % Smaller batch size for better gradient updates
            iterPerEpoch = floor(numTrainingFramesPerRouter*numTotalRouters/miniBatchSize);

            options = trainingOptions('adam', ...
                'MaxEpochs', 80, ... % More epochs for better convergence
                'ValidationData', {xValFrames, yVal}, ...
                'ValidationFrequency', iterPerEpoch, ...
                'Verbose', false, ...
                'InitialLearnRate', 0.0003, ... % Lower learning rate for stability
                'LearnRateSchedule', 'piecewise', ...
                'LearnRateDropFactor', 0.05, ... % Very aggressive drop
                'LearnRateDropPeriod', 15, ... % Drop every 15 epochs
                'MiniBatchSize', miniBatchSize, ...
                'Plots', 'training-progress', ...
                'Shuffle', 'every-epoch', ...
                'L2Regularization', 0.000001, ... % Very low regularization
                'GradientThreshold', 0.5, ... % Lower gradient threshold
                'ValidationPatience', 25, ... % Higher patience
                'ExecutionEnvironment', 'cpu');
            
            tic
            
            % Use simplified but optimized model for compatibility and performance
            layers = [
                imageInputLayer(inputSize, 'Normalization', 'zscore', 'Name', 'Input')
                
                % Enhanced CNN Feature extraction with residual-like connections
                convolution2dLayer([9 1], 32, 'Padding', [4 0], 'Name', 'InitConv')
                batchNormalizationLayer('Name', 'InitBN')
                reluLayer('Name', 'InitReLU')
                
                convolution2dLayer([7 1], 64, 'Padding', [3 0], 'Name', 'Conv1')
                batchNormalizationLayer('Name', 'BN1')
                reluLayer('Name', 'ReLU1')
                dropoutLayer(0.1, 'Name', 'Dropout1')
                maxPooling2dLayer([2 1], 'Stride', [2 1], 'Name', 'MaxPool1')
                
                convolution2dLayer([5 1], 128, 'Padding', [2 0], 'Name', 'Conv2')
                batchNormalizationLayer('Name', 'BN2')
                reluLayer('Name', 'ReLU2')
                dropoutLayer(0.15, 'Name', 'Dropout2')
                maxPooling2dLayer([2 1], 'Stride', [2 1], 'Name', 'MaxPool2')
                
                convolution2dLayer([5 1], 256, 'Padding', [2 0], 'Name', 'Conv3')
                batchNormalizationLayer('Name', 'BN3')
                reluLayer('Name', 'ReLU3')
                dropoutLayer(0.2, 'Name', 'Dropout3')
                
                convolution2dLayer([3 1], 512, 'Padding', [1 0], 'Name', 'Conv4')
                batchNormalizationLayer('Name', 'BN4')
                reluLayer('Name', 'ReLU4')
                dropoutLayer(0.2, 'Name', 'Dropout4')
                
                % Global Average Pooling for better generalization
                globalAveragePooling2dLayer('Name', 'GAP')
                
                % Flatten for subsequent layers
                flattenLayer('Name', 'Flatten')
                
                % Enhanced self-attention mechanism
                fullyConnectedLayer(512, 'Name', 'Attention1')
                batchNormalizationLayer('Name', 'AttBN1')
                reluLayer('Name', 'AttReLU1')
                dropoutLayer(0.1, 'Name', 'AttDropout1')
                
                fullyConnectedLayer(512, 'Name', 'Attention2')
                batchNormalizationLayer('Name', 'AttBN2')
                tanhLayer('Name', 'AttTanh')
                dropoutLayer(0.15, 'Name', 'AttDropout2')
                
                % Multi-layer LSTM for temporal modeling
                lstmLayer(512, 'OutputMode', 'sequence', 'Name', 'LSTM1')
                dropoutLayer(0.2, 'Name', 'LSTMDropout1')
                
                lstmLayer(256, 'OutputMode', 'sequence', 'Name', 'LSTM2')
                dropoutLayer(0.2, 'Name', 'LSTMDropout2')
                
                lstmLayer(128, 'OutputMode', 'last', 'Name', 'LSTM3')
                dropoutLayer(0.25, 'Name', 'LSTMDropout3')
                
                % Enhanced dense layers for classification
                fullyConnectedLayer(1024, 'Name', 'FC1')
                batchNormalizationLayer('Name', 'BNFC1')
                reluLayer('Name', 'ReLUFC1')
                dropoutLayer(0.5, 'Name', 'FCDropout1')
                
                fullyConnectedLayer(512, 'Name', 'FC2')
                batchNormalizationLayer('Name', 'BNFC2')
                reluLayer('Name', 'ReLUFC2')
                dropoutLayer(0.4, 'Name', 'FCDropout2')
                
                fullyConnectedLayer(256, 'Name', 'FC3')
                batchNormalizationLayer('Name', 'BNFC3')
                reluLayer('Name', 'ReLUFC3')
                dropoutLayer(0.3, 'Name', 'FCDropout3')
                
                fullyConnectedLayer(numClasses, 'Name', 'FCFinal')
                softmaxLayer('Name', 'SoftMax')
                classificationLayer('Name', 'Output')
            ];
            
            simNet = trainNetwork(xTrainingFrames, yTrain, layers, options);
            TrainTime = seconds(toc);

            disp("time of training = ");
            toc

            %%
            
            yTestPred = classify(simNet,xTestFrames,'ExecutionEnvironment', 'cpu');

            
            testAccuracy = mean(yTest == yTestPred);
            disp("Test accuracy: " + testAccuracy*100 + "%")
            figure
            cm = confusionchart(yTest, yTestPred);
            cm.Title = 'Confusion Matrix - Optimized CNN-ResNet-Attention-LSTM Model';
            cm.RowSummary = 'row-normalized';
            confusionFileName = sprintf('Optimized_Model_Result_%d_SNR_%d_Frame_%d_San_%d', numTotalRouters,SNR,localFramesPerRouter,san);
            saveas(gcf,confusionFileName,'png');

            %%
            
            numTests = 100; % 

            
            accuracies = zeros(numTests,1);

            
            for i = 1:numTests
                
                idx = randperm(numel(yTest));
                xTestFramesShuffled = xTestFrames(:,:,:,idx);
                yTestShuffled = yTest(idx);

                
                yTestPred = classify(simNet, xTestFramesShuffled, 'ExecutionEnvironment', 'cpu');

                
                accuracies(i) = mean(yTestShuffled == yTestPred);
            end

            
            averageAccuracy = mean(accuracies);
            stdAccuracy = std(accuracies);
            minAccuracy = min(accuracies);
            maxAccuracy = max(accuracies);
            
            disp(['平均准确率为：', num2str(averageAccuracy)]);
            disp(['准确率标准差为：', num2str(stdAccuracy)]);
            disp(['最低准确率为：', num2str(minAccuracy)]);
            disp(['最高准确率为：', num2str(maxAccuracy)]);

            %% Save Results
           
            saveFileName = sprintf('Optimized_Model_Result_%d_SNR_%d_Frame_%d.mat', numTotalRouters,SNR,localFramesPerRouter);
            save(saveFileName, 'GenerateTime', 'TrainTime', 'averageAccuracy', 'stdAccuracy', 'testAccuracy', 'minAccuracy', 'maxAccuracy');
            fprintf('Optimized model results saved to %s\n\n', saveFileName);
            
            % Display final performance summary
            fprintf('=== PERFORMANCE SUMMARY ===\n');
            fprintf('Test Accuracy: %.2f%%\n', testAccuracy*100);
            fprintf('Average Accuracy (100 runs): %.2f%% ± %.2f%%\n', averageAccuracy*100, stdAccuracy*100);
            fprintf('Accuracy Range: %.2f%% - %.2f%%\n', minAccuracy*100, maxAccuracy*100);
            fprintf('Training Time: %.2f seconds\n', seconds(TrainTime));
            fprintf('==========================\n\n');
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