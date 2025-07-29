% Enhanced RF Fingerprinting with CNN + ResNet + Self-Attention + LSTM
% Version 2: Improved architecture with proper ResNet skip connections

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

            %%
            % Enhanced Model Parameters
            inputSize = [frameLength 2 1]; 
            numHiddenUnits = 128; 
            numClasses = numKnownRouters + 1; 
            
            % Enhanced architecture parameters
            attentionDim = 128;  % Self-attention dimension
            
            % Create LayerGraph for Complex Architecture
            lgraph = layerGraph();
            
            % Input Layer
            tempLayers = [
                imageInputLayer(inputSize, 'Normalization', 'none', 'Name', 'input')
                convolution2dLayer([8, 2], 32, 'Padding', 'same', 'Name', 'initial_conv')
                batchNormalizationLayer('Name', 'initial_bn')
                reluLayer('Name', 'initial_relu')
            ];
            lgraph = addLayers(lgraph, tempLayers);
            
            % ResNet Block 1 - Main Path
            tempLayers = [
                convolution2dLayer([3, 1], 32, 'Padding', 'same', 'Name', 'resnet1_conv1')
                batchNormalizationLayer('Name', 'resnet1_bn1')
                reluLayer('Name', 'resnet1_relu1')
                convolution2dLayer([3, 1], 32, 'Padding', 'same', 'Name', 'resnet1_conv2')
                batchNormalizationLayer('Name', 'resnet1_bn2')
            ];
            lgraph = addLayers(lgraph, tempLayers);
            
            % ResNet Block 1 - Skip Connection
            tempLayers = [
                additionLayer(2, 'Name', 'resnet1_add')
                reluLayer('Name', 'resnet1_output')
                maxPooling2dLayer([2, 1], 'Stride', [2, 1], 'Name', 'resnet1_pool')
            ];
            lgraph = addLayers(lgraph, tempLayers);
            
            % Connect ResNet Block 1
            lgraph = connectLayers(lgraph, 'initial_relu', 'resnet1_conv1');
            lgraph = connectLayers(lgraph, 'initial_relu', 'resnet1_add/in2');
            lgraph = connectLayers(lgraph, 'resnet1_bn2', 'resnet1_add/in1');
            lgraph = connectLayers(lgraph, 'resnet1_add', 'resnet1_output');
            lgraph = connectLayers(lgraph, 'resnet1_output', 'resnet1_pool');
            
            % ResNet Block 2 - Main Path
            tempLayers = [
                convolution2dLayer([3, 1], 64, 'Padding', 'same', 'Name', 'resnet2_conv1')
                batchNormalizationLayer('Name', 'resnet2_bn1')
                reluLayer('Name', 'resnet2_relu1')
                convolution2dLayer([3, 1], 64, 'Padding', 'same', 'Name', 'resnet2_conv2')
                batchNormalizationLayer('Name', 'resnet2_bn2')
            ];
            lgraph = addLayers(lgraph, tempLayers);
            
            % ResNet Block 2 - Skip Connection (with 1x1 conv for dimension matching)
            tempLayers = [
                convolution2dLayer([1, 1], 64, 'Padding', 'same', 'Name', 'resnet2_skip_conv')
                batchNormalizationLayer('Name', 'resnet2_skip_bn')
            ];
            lgraph = addLayers(lgraph, tempLayers);
            
            tempLayers = [
                additionLayer(2, 'Name', 'resnet2_add')
                reluLayer('Name', 'resnet2_output')
                maxPooling2dLayer([2, 1], 'Stride', [2, 1], 'Name', 'resnet2_pool')
            ];
            lgraph = addLayers(lgraph, tempLayers);
            
            % Connect ResNet Block 2
            lgraph = connectLayers(lgraph, 'resnet1_pool', 'resnet2_conv1');
            lgraph = connectLayers(lgraph, 'resnet1_pool', 'resnet2_skip_conv');
            lgraph = connectLayers(lgraph, 'resnet2_bn2', 'resnet2_add/in1');
            lgraph = connectLayers(lgraph, 'resnet2_skip_bn', 'resnet2_add/in2');
            lgraph = connectLayers(lgraph, 'resnet2_add', 'resnet2_output');
            lgraph = connectLayers(lgraph, 'resnet2_output', 'resnet2_pool');
            
            % Enhanced Feature Extraction
            tempLayers = [
                convolution2dLayer([4, 1], 128, 'Padding', 'same', 'Name', 'enhanced_conv1')
                batchNormalizationLayer('Name', 'enhanced_bn1')
                reluLayer('Name', 'enhanced_relu1')
                dropoutLayer(0.3, 'Name', 'enhanced_dropout1')
                
                convolution2dLayer([4, 1], 256, 'Padding', 'same', 'Name', 'enhanced_conv2')
                batchNormalizationLayer('Name', 'enhanced_bn2')
                reluLayer('Name', 'enhanced_relu2')
                dropoutLayer(0.3, 'Name', 'enhanced_dropout2')
                
                % Global Average Pooling
                globalAveragePooling2dLayer('Name', 'global_avg_pool')
                
                % Self-Attention Mechanism (Simplified)
                fullyConnectedLayer(attentionDim, 'Name', 'attention_query')
                reluLayer('Name', 'attention_query_relu')
                fullyConnectedLayer(attentionDim, 'Name', 'attention_key')
                reluLayer('Name', 'attention_key_relu')
                fullyConnectedLayer(attentionDim, 'Name', 'attention_value')
                reluLayer('Name', 'attention_value_relu')
                
                % Attention Output
                fullyConnectedLayer(attentionDim, 'Name', 'attention_output')
                reluLayer('Name', 'attention_output_relu')
                dropoutLayer(0.4, 'Name', 'attention_dropout')
                
                % Prepare for LSTM
                fullyConnectedLayer(numHiddenUnits, 'Name', 'pre_lstm_fc')
                reluLayer('Name', 'pre_lstm_relu')
                
                % LSTM Layers
                lstmLayer(numHiddenUnits, 'OutputMode', 'sequence', 'Name', 'lstm1')
                dropoutLayer(0.5, 'Name', 'lstm_dropout1')
                
                lstmLayer(numHiddenUnits, 'OutputMode', 'last', 'Name', 'lstm2')
                dropoutLayer(0.5, 'Name', 'lstm_dropout2')
                
                % Final Classification
                fullyConnectedLayer(numHiddenUnits/2, 'Name', 'pre_classification_fc')
                reluLayer('Name', 'pre_classification_relu')
                dropoutLayer(0.3, 'Name', 'pre_classification_dropout')
                
                fullyConnectedLayer(numClasses, 'Name', 'classification_fc')
                softmaxLayer('Name', 'softmax')
                classificationLayer('Name', 'classification_output')
            ];
            lgraph = addLayers(lgraph, tempLayers);
            
            % Connect final layers
            lgraph = connectLayers(lgraph, 'resnet2_pool', 'enhanced_conv1');
            
            % Enhanced Training Options
            miniBatchSize = 256;
            iterPerEpoch = floor(numTrainingFramesPerRouter*numTotalRouters/miniBatchSize);
            
            options = trainingOptions('adam', ...
                'MaxEpochs', 30, ...
                'ValidationData', {xValFrames, yVal}, ...
                'ValidationFrequency', iterPerEpoch, ...
                'Verbose', false, ...
                'InitialLearnRate', 0.001, ...
                'LearnRateSchedule', 'piecewise', ...
                'LearnRateDropFactor', 0.3, ...
                'LearnRateDropPeriod', 5, ...
                'MiniBatchSize', miniBatchSize, ...
                'Plots', 'training-progress', ...
                'Shuffle', 'every-epoch', ...
                'L2Regularization', 0.0015, ...
                'GradientThreshold', 1, ...
                'ExecutionEnvironment', 'cpu');
            
            % Model Training
            tic
            fprintf('Training Enhanced CNN-ResNet-Attention-LSTM Model v2...\n');
            simNet = trainNetwork(xTrainingFrames, yTrain, lgraph, options);
            TrainTime = seconds(toc);

            disp("Enhanced model v2 training time = ");
            toc

            %%
            % Model Testing
            fprintf('Testing Enhanced Model v2...\n');
            yTestPred = classify(simNet,xTestFrames,'ExecutionEnvironment', 'cpu');

            testAccuracy = mean(yTest == yTestPred);
            disp("Enhanced Model v2 Test accuracy: " + testAccuracy*100 + "%")
            
            figure
            cm = confusionchart(yTest, yTestPred);
            cm.Title = 'Enhanced Model v2 Confusion Matrix for Test Data';
            cm.RowSummary = 'row-normalized';
            confusionFileName = sprintf('Enhanced_v2_Result_%d_SNR_%d_Frame_%d_San_%d', numTotalRouters,SNR,localFramesPerRouter,san);
            saveas(gcf,confusionFileName,'png');

            %%
            % Multiple Test Runs for Statistical Analysis
            numTests = 100; 
            accuracies = zeros(numTests,1);

            fprintf('Running %d statistical tests...\n', numTests);
            for i = 1:numTests
                idx = randperm(numel(yTest));
                xTestFramesShuffled = xTestFrames(:,:,:,idx);
                yTestShuffled = yTest(idx);

                yTestPred = classify(simNet, xTestFramesShuffled, 'ExecutionEnvironment', 'cpu');
                accuracies(i) = mean(yTestShuffled == yTestPred);
                
                if mod(i, 20) == 0
                    fprintf('Completed %d/%d tests\n', i, numTests);
                end
            end

            averageAccuracy = mean(accuracies);
            stdAccuracy = std(accuracies);
            
            disp(['Enhanced Model v2 - Average Accuracy: ', num2str(averageAccuracy*100), '%']);
            disp(['Enhanced Model v2 - Standard Deviation: ', num2str(stdAccuracy*100), '%']);

            %% Save Enhanced Results
            saveFileName = sprintf('Enhanced_v2_Result_%d_SNR_%d_Frame_%d.mat', numTotalRouters,SNR,localFramesPerRouter);
            save(saveFileName, 'GenerateTime', 'TrainTime', 'averageAccuracy', 'stdAccuracy', 'accuracies');
            fprintf('Enhanced v2 results saved to %s\n\n', saveFileName);
        end
    end
end

%%
% Helper Functions (unchanged from original)
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