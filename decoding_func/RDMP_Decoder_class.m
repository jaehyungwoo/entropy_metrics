classdef RDMP_Decoder_class
    properties 
        TrainedDecoder   % cell-array trained decoder
        TrainType        % Training type, either moving window ('MovWin') or across-trial ('Trial')
        MetricType       % metric computation type; difference from first window or raw values
        config           % contains all configurations
    end

    methods (Access = public)
        % Constructor
        function obj = RDMP_Decoder_class(DecodeOpt)
            obj.TrainedDecoder = {};
            obj.TrainType = DecodeOpt.train_type;
            obj.config = DecodeOpt;
            poss_type = ["MovWin", "Trial"];
            assert(ismember(DecodeOpt.train_type, poss_type),"Training type can be: "+strjoin(poss_type," / "));
            
            poss_type = ["pessimistic", "neutral"];
            assert(ismember(DecodeOpt.comp_group, poss_type),"Comparison group should be: "+strjoin(poss_type," / "));
            
            poss_type = ["grid", "random"];
            assert(ismember(DecodeOpt.sample_type, poss_type),"Sampling options should be: "+strjoin(poss_type," / "));
    
            if ~isfield(obj.config,'learner')
                obj.config.learner = "logistic";
            end
            if ~isfield(obj.config,'metric_type')
                obj.config.metric_type = "raw";
            end
            obj.MetricType = obj.config.metric_type;
            disp("Decoder class initialized: "+DecodeOpt.train_type+", "+obj.config.metric_type+" values");
        end
        
        %% training function
        function [obj, AllGroupDat] = train_decoder(obj, alpha_range, stepSize, beta1, Sd)
            % RewProbs: trial-by-trial reward probabilites of two options in each column
            % beta1: fixed inv. temp. of models
            % stepSize: step size for sampling alpha's
            % Sd: random seed for reproducibility
            
            % Sample alpha values for simulating
            disp("alpha step size: "+stepSize);            
            alph_set = alpha_range(1,1):stepSize:alpha_range(1,2);
            
            switch obj.config.sample_type
                case "grid"
                    DeltaAlpha = flip(alph_set') - alph_set;
                    [X, Y] = meshgrid(alph_set, flip(alph_set)); % x = alpha-, y = alpha+    
                    minDiff = stepSize*1.5;
                    alphas = [X(DeltaAlpha>minDiff), Y(DeltaAlpha>minDiff)];      
                    
                    rng(Sd);
                    opt_alphas = alphas + rand(size(alphas))*stepSize/2; % add random jitters
                    pes_alphas = flip(opt_alphas, 2);
                    
                    % neut_alphas = linspace(alpha_range(1,1), alpha_range(1,2), size(opt_alphas,1));
                    neut_alphas = mean(alpha_range) + (rand(size(opt_alphas,1),1)-0.5)*stepSize;
                case "random"
                    % random sample from specified range?
                    rng(Sd);
                    neut_alphas = alpha_range(1,1) + diff(alpha_range(1,:))*rand(length(alph_set),1);
                    opt_alphas(:,2) = neut_alphas;
                    opt_alphas(:,1) = opt_alphas(:,2) .* rand(size(neut_alphas));
                    pes_alphas = alpha_range(1,1) + diff(alpha_range(1,:))*rand(length(alph_set),1);
                    pes_alphas(:,2) = pes_alphas .* rand(size(neut_alphas));
            end
            agentN = size(opt_alphas,1);
            disp(agentN+" sample points");

            func_hand = str2func(obj.config.train_type+"_train_decoder"); 
            [obj, AllGroupDat] = func_hand(obj, opt_alphas, pes_alphas, neut_alphas, beta1);
        end
        
        %% simulate testing data: RL1, RDMP, RL2
        function [ModOutput] = obtain_test_data(obj, params)
            func_hand = str2func("obtain_"+obj.config.train_type+"_metrics_RDMP");            
            [ModOutput, ~] = func_hand(obj, params);
        end

        %% if cross-validating within dataset
        % train label: RDMP or RL2
        function [PostProb, AccuMAT] = cross_validate_data(obj, ModOutput)                        
            CrossDat = ModOutput(2:3);

            func_hand = str2func("crossval_"+obj.config.train_type+"_metrics");
            [PostProb, AccuMAT] = func_hand(obj, CrossDat);
        end

        %% testing function
        function [PostProb, AccuMAT] = test_decoder(obj, ModOutput)

            func_hand = str2func("test_decoder_"+obj.config.train_type+"_metrics");
            [PostProb, AccuMAT] = func_hand(obj, ModOutput);
        end
    end
    
    methods (Access = private)
        %% training decoder on the grid of alpha's
        function [obj, AllGroupDat] = MovWin_train_decoder(obj, opt_alphas, pes_alphas, neut_alphas, beta1)
            DecodeVar_set = obj.config.VarSet;   
            windowL = obj.config.windowL;    
            withinN = obj.config.withinN;
            RewProbs = obj.config.RewProbs;
            
            comparison_group = obj.config.comp_group;

            if contains(obj.config.metric_type,"diff")
                diff_flag = 1;
            else
                diff_flag = 0;
            end

            blockL = size(RewProbs, 1);
            agentN = size(opt_alphas,1);
    
            disp("Trained on : optimistic vs. "+comparison_group);
            assert(strcmp(comparison_group,"neutral")||strcmp(comparison_group,"pessimistic"));
            disp("WithinN = "+withinN);

             % intialize model
            SimMod = struct;
            SimMod.name = 'RL_2alpha';      
            SimMod.fun = 'simRL_2alpha'; % specify simulation function   
            SimMod.initpar =[.5   5 .5];       
            SimMod.lb      =[ 0   1  0];   
            SimMod.ub      =[ 1 100  1];   
        
            DecompMap = obj.intialize_decomp_map();
            R_decomp  = DecompMap.R;
            O_decomp  = DecompMap.O;
            RO_decomp = DecompMap.RO;

            %% Decoder training: Run simulation and compute moving-window metrics
            simStart = tic;
            AllGroupDat = struct;
            for group = ["optimistic", comparison_group]                        
                %% loop through each agent
                clear agentMet 
                agentMet(agentN) = struct;
                parfor n = 1:agentN
                    alpha_plus = []; alpha_minus = []; % clear par vars
                    switch group
                        case "optimistic"
                            alpha_plus  = opt_alphas(n,2); % note this is y-coord in the grid
                            alpha_minus = opt_alphas(n,1); assert(alpha_plus>alpha_minus);
                        case "neutral"
                            alpha_plus  = neut_alphas(n);
                            alpha_minus = neut_alphas(n);
                        case "pessimistic"
                            alpha_plus  = pes_alphas(n,2);
                            alpha_minus = pes_alphas(n,1); assert(alpha_plus<alpha_minus);
                    end

                    % cap alpha b/w [0, 1]
                    alpha_plus  = max(0, min(alpha_plus,1));
                    alpha_minus = max(0, min(alpha_minus,1));
                    %% generate behavior w/ each sample of (alpha-, alpha+)
                    % agentMet(n).label_group = repmat(group, 1, 1);
        
                    % set up environment and agent
                    simEnv = SetUp_RewardEnvironment(RewProbs, n);                         
                    player = struct;
                    player.label  = 'simRL_2alpha';
                    player.params = [alpha_plus, beta1, alpha_minus];

                    % within-sample averages?
                    tempTrials = struct;
                    for nn = 1:withinN
                        simStats = Simulate_ModelChoice_randR(player, simEnv, 'AB',(n-1)*agentN + nn); 
                        stay  = simStats.c(1:end-1)==simStats.c(2:end);
                        prevR = simStats.r(1:end-1);
                        prevO = simStats.c(1:end-1)==simEnv.hr_stim(1:end-1);
        
                        % compute moving-window metrics through trials
                        for t = windowL:blockL-1
                            trial_idx = t - windowL + 1:t;
                            tempTrials = obj.compute_and_append_metrics(tempTrials, {stay(trial_idx), prevR(trial_idx), prevO(trial_idx)});
                        end                
                    end
                    Fi = fieldnames(tempTrials);
                    for f = 1:length(Fi)                
                        agentMet(n).(Fi{f}) = mean(reshape(tempTrials.(Fi{f}), numel(windowL:blockL-1), []), 2, 'omitnan'); % trials-by-1 vec
                        if diff_flag
                            agentMet(n).(Fi{f})(isnan(agentMet(n).(Fi{f}))) = 0; % rectify all NaN's to zeros?
                            agentMet(n).(Fi{f}) = agentMet(n).(Fi{f}) - agentMet(n).(Fi{f})(1)
                        end
                    end
                end
                tempGroup = struct;
                Fi = fieldnames(agentMet);
                for f = 1:length(Fi)
                    tempGroup.(Fi{f}) = [agentMet.(Fi{f})]';
                end
                tempGroup.label_group = repmat(group, size(tempGroup.(Fi{f}),1), 1);
                AllGroupDat = append_to_fields(AllGroupDat, {tempGroup});
            end
            disp("Sim complete: "+toc(simStart)/60+ "min");
            
            %% Train decoders from obtained samples & labels 
            trainStart = tic;
        
            T = struct2table(AllGroupDat);
            if obj.config.conv_nan2zero
                % convert NaN into zero?
                numVars = vartype('numeric');
                T(:, numVars) = fillmissing(T(:, numVars), 'constant', 0);
            end
            T_clean = T(~any(ismissing(T), 2), :);
            disp("    Data length = "+height(T_clean));
        
            %%% Loop through each set of features and train decoders for each running window    
            numTrials = numel(windowL:blockL-1);
            DecoderTrials = cell(length(DecodeVar_set), numTrials); % cell array of trained decoders for each window
            
            for f_set = 1:length(DecodeVar_set)
                met_to_keep = DecodeVar_set{f_set};
                if strcmp(DecodeVar_set{f_set}, "all")
                    met_to_keep = setdiff(string(fieldnames(AllGroupDat)), "label_group");            
                end
                T_reduced = T_clean(:, met_to_keep);
        
                for tt = 1:numTrials
                    T_window = varfun(@(x) x(:,tt), T_reduced);
                    T_window.Properties.VariableNames = T_reduced.Properties.VariableNames;           
                    
                    % add labels (response var)
                    T_window.label_group = (T_clean.label_group=="optimistic");
        
                    % train decoder (logistic)
                    DecoderTrials{f_set, tt} = fitclinear(T_window, 'label_group', 'Learner',obj.config.learner);
                end
            end
            disp("Decoder training complete: "+toc(trainStart)+" sec");

            obj.TrainedDecoder = DecoderTrials;
        end

        %% Trial version
        function [obj, AllGroupDat] = Trial_train_decoder(obj, opt_alphas, pes_alphas, neut_alphas, beta1)
            DecodeVar_set = obj.config.VarSet;    
            RewProbs = obj.config.RewProbs;
            numEnv = obj.config.numEnv;
            numSim = obj.config.numSim;

            comparison_group = obj.config.comp_group;

            if contains(obj.config.metric_type,"diff")
                diff_flag = 1;
            else
                diff_flag = 0;
            end

            blockL = size(RewProbs, 1);
            agentN = size(opt_alphas,1);
    
            disp("Trained on : optimistic vs. "+comparison_group);
            assert(strcmp(comparison_group,"neutral")||strcmp(comparison_group,"pessimistic"));
        
            DecompMap = obj.intialize_decomp_map();
            R_decomp  = DecompMap.R;
            O_decomp  = DecompMap.O;
            RO_decomp = DecompMap.RO;

            %% Decoder training: Run simulation and compute aross-trial metrics
            simStart = tic;
            AllGroupDat = struct;
            for group = ["optimistic", comparison_group]                        
                %% loop through each agent
                clear agentMet 
                agentMet(agentN) = struct;
                parfor n = 1:agentN
                    if mod(n,10)==0; disp(n+"/"+agentN); end
                    alpha_plus = []; alpha_minus = []; % clear par vars
                    switch group
                        case "optimistic"
                            alpha_plus  = opt_alphas(n,2); % note this is y-coord in the grid
                            alpha_minus = opt_alphas(n,1); assert(alpha_plus>alpha_minus);
                        case "neutral"
                            alpha_plus  = neut_alphas(n);
                            alpha_minus = neut_alphas(n);
                        case "pessimistic"
                            alpha_plus  = pes_alphas(n,2);
                            alpha_minus = pes_alphas(n,1); assert(alpha_plus<alpha_minus);
                    end
                    % cap alpha b/w [0, 1]
                    alpha_plus  = max(0, min(alpha_plus,1));
                    alpha_minus = max(0, min(alpha_minus,1));
                    %% generate behavior w/ each sample of (alpha-, alpha+)
                    tempTrials = struct;
                    for ne = 1:numEnv
                        % set up environment and agent
                        simEnv = SetUp_RewardEnvironment(RewProbs, ne);                         
                        player = struct;
                        player.label  = 'simRL_2alpha';
                        player.params = [alpha_plus, beta1, alpha_minus];
            
                        % tempTrials = struct;
                        CompStat = struct;
                        for ns = 1:numSim
                            simStats = Simulate_ModelChoice_randR(player, simEnv, 'AB',(ne-1)*numSim + ns); 
                            stay  = simStats.c(1:end-1)==simStats.c(2:end);
                            prevR = simStats.r(1:end-1);
                            prevO = simStats.c(1:end-1)==simEnv.hr_stim(1:end-1);
            
                            tempStats = struct;
                            tempStats.stay = stay';
                            tempStats.prevR = prevR';
                            tempStats.prevO = prevO';
            
                            CompStat = append_to_fields(CompStat, {tempStats});
                        end
                        
                        % calculate metric for each trial position                
                        for t = 1:blockL-1
                            stay  = CompStat.stay(:,t);             
                            prevR = CompStat.prevR(:,t);
                            prevO = CompStat.prevO(:,t);     
                            prevRO = binary_to_decimal([prevR, prevO]);

                            tempH = struct;
                            tempH.H_str = Shannon_Entropy(stay);
                            tempTrials = append_to_fields(tempTrials, {tempH, Conditional_Entropy_decomp(stay, prevR, "ERDS", R_decomp), ...
                                                    Mutual_Information_decomp(stay, prevR, "MIRS", R_decomp), ...
                                                    Conditional_Entropy_decomp(stay, prevO, "EODS", O_decomp ), ...
                                                    Mutual_Information_decomp(stay, prevO, "MIOS", O_decomp), ...
                                                    Conditional_Entropy_decomp(stay, prevRO, "ERODS", RO_decomp), ...
                                                    Mutual_Information_decomp(stay, prevRO, "MIROS", RO_decomp) });
                        end   
                    end
        
                    % take 'theoretical averages'
                    Fi = fieldnames(tempTrials);
                    for f = 1:length(Fi)                
                        agentMet(n).(Fi{f}) = mean(reshape(tempTrials.(Fi{f}), [], numEnv), 2, 'omitnan'); % trials-by-1 vec
                    end
                end
                tempGroup = struct;
                Fi = fieldnames(agentMet);
                for f = 1:length(Fi)
                    tempGroup.(Fi{f}) = reshape([agentMet.(Fi{f})], blockL-1, [])';
                end
                tempGroup.label_group = repmat(group, size(tempGroup.ERDS,1), 1);
                AllGroupDat = append_to_fields(AllGroupDat, {tempGroup});
            end
            disp("Sim complete: "+toc(simStart)/60+ "min");
            %% Train decoders from obtained samples & labels 
            trainStart = tic;
        
            T = struct2table(AllGroupDat);
            if obj.config.conv_nan2zero
                % convert NaN into zero?
                numVars = vartype('numeric');
                T(:, numVars) = fillmissing(T(:, numVars), 'constant', 0);
            end
            T_clean = T(~any(ismissing(T), 2), :);
            disp("    Data length = "+height(T_clean));
        
            %%% Loop through each set of features and train decoders for each running window    
            numTrials = blockL - 1;
            DecoderTrials = cell(length(DecodeVar_set), numTrials); % cell array of trained decoders for each window
            
            for f_set = 1:length(DecodeVar_set)
                met_to_keep = DecodeVar_set{f_set};
                if strcmp(DecodeVar_set{f_set}, "all")
                    met_to_keep = setdiff(string(fieldnames(AllGroupDat)), "label_group");            
                end
                T_reduced = T_clean(:, met_to_keep);
        
                for tt = 1:numTrials
                    T_window = varfun(@(x) x(:,tt), T_reduced);
                    T_window.Properties.VariableNames = T_reduced.Properties.VariableNames;           
                    
                    % add labels (response var)
                    T_window.label_group = (T_clean.label_group=="optimistic");
        
                    % train decoder (logistic)
                    DecoderTrials{f_set, tt} = fitclinear(T_window, 'label_group', 'Learner', 'logistic');
                end
            end
            disp("Decoder training complete: "+toc(trainStart)+" sec");

            obj.TrainedDecoder = DecoderTrials;
        end

        %% obtain test data: simulated metrics from models
        function [MovMetOut, models] = obtain_MovWin_metrics_RDMP(obj, params)
            % simulate metrics from three models: RL1, RDMP, and RL2 (fitted to RDMP)
            % first simulate the first two models
            % then fit the third RL2 model to RDMP and simulate
        
            windowL = obj.config.windowL;
            RewProbs = obj.config.RewProbs;
            blockL = size(RewProbs, 1);
            numTrials = numel(windowL:blockL-1);

            numSim = obj.config.numSim;
            numEnv = obj.config.numEnv;
            sessfit_flag = obj.config.sessfit_flag;

            if contains(obj.config.metric_type,"diff")
                diff_flag = 1;
            else
                diff_flag = 0;
            end
            %% Initialize models    
            alph0   = params(1);
            beta1   = params(2);
            mp1     = params(3);
            
            models = obj.init_sim_models([alph0, beta1, mp1, alph0]);
        
            fname = "MovWinMet"+windowL+"_"+(length(models)+1)+"mods_a"+alph0+"_b"+beta1+"_m"+mp1+"_n"+numEnv+"x"+numSim;
            if sessfit_flag
                fname = fname + "_sessFit";
            end
            
            if exist(fname+".mat",'file')  
                load("output/"+fname+".mat", 'MovMetOut','models');
                disp("File exists, loading simulation output: "+fname);
                return;
            else
                disp("Running simulation of models...")
            end
           
            %% Simulate moving window metrics from first two models
            MovMetOut = cell(1, length(models)+1);
            for m = 1:length(models)
                player = struct;
                player.label = models{m}.fun;
                player.params = models{m}.simpar; 
                
                clear EntTrial EffAlphas
                EntTrial(numEnv) = struct; % across trials w.r.t. rev 
                EffAlphas(numEnv) = struct;
        
                %%
                parfor ne = 1:numEnv              
                    CompStat = struct;
                    CompStat.eff_plus = []; 
                    CompStat.eff_minus = [];
        
                    simEnv = SetUp_RewardEnvironment(RewProbs, ne);
                    %% loop through repeated blocks for this env.  
                    tempTrials = struct;
                    for ns = 1:numSim
                        simStats = Simulate_ModelChoice_randR(player, simEnv, 'AB', ns+(ne-1)*numSim);
                        stay  = simStats.c(1:end-1)==simStats.c(2:end);
                        prevR = simStats.r(1:end-1); 
                        prevO = simStats.c(1:end-1)==simEnv.hr_stim(1:end-1);
                        % prevRO = binary_to_decimal([prevR, prevO]);
                        
                        if contains(player.label,"RDMP")
                            Q = [simStats.q1, simStats.q2];
                            Q_c = Q(sub2ind(size(Q), (1:blockL)', (simStats.c/2)+1.5)); % Q_chosen
                            DeltaQ = [diff(Q,[],1); nan(1,2)];
                            DeltaQc = DeltaQ(sub2ind(size(Q), (1:blockL)', (simStats.c/2)+1.5)); % deltaQ_chosen                
                            eff_alpha = DeltaQc ./ (simStats.r - Q_c); % effective learning rate    
                            eff_plus  = eff_alpha; eff_plus(simStats.r==0) = NaN;
                            eff_minus = eff_alpha; eff_minus(simStats.r==1) = NaN;
                            CompStat.eff_plus = [CompStat.eff_plus; eff_plus'];
                            CompStat.eff_minus = [CompStat.eff_minus; eff_minus'];
                        end
                        % compute moving-window metrics through trials                
                        for t = windowL:blockL-1
                            trial_idx = t-windowL+1:t;
                            tempTrials = obj.compute_and_append_metrics(tempTrials, {stay(trial_idx), prevR(trial_idx), prevO(trial_idx)});
                        end
                    end
                    Fi = fieldnames(tempTrials);
                    for f = 1:length(Fi)
                        EntTrial(ne).(Fi{f}) = tempTrials.(Fi{f}); % 1-by-trials MAT
                    end
                    
                    % store effective learn. rates
                    if contains(player.label,"RDMP")
                        EffAlphas(ne).eff_plus  = mean(CompStat.eff_plus , 1,'omitnan');
                        EffAlphas(ne).eff_minus = mean(CompStat.eff_minus, 1,'omitnan');
                    end
                end
                % reshape output data        
                Fi = fieldnames(EntTrial);
                for f = 1:length(Fi)
                    MovMetOut{m}.(Fi{f}) = reshape([EntTrial.(Fi{f})], numTrials,[])'; % 1-by-trials MAT
                end
        
                if contains(player.label,"RDMP")
                    MovMetOut{m}.eff_plus  = reshape([EffAlphas.eff_plus], [], numEnv)';
                    MovMetOut{m}.eff_minus = reshape([EffAlphas.eff_minus], [], numEnv)';
                end
            end
            
            %% Fit RL2 to choice behavior of RDMP, and obtain moving window metrics
            fStart = tic;
            fit_beta_flag = 0;
        
            % optimize options
            numFit = 5;
            op = optimset('fminsearch');
            op.MaxIter = 1e7; op.MaxFunEvals = 1e7;
           
            clear EntTrial EffAlphas
            EntTrial(numEnv) = struct; % across trials w.r.t. rev 
            EffAlphas(numEnv) = struct;
            parfor ne = 1:numEnv
                fpar = [];
                FitMod = models{1}; % RL2
                SimMod = models{2}; % RDMP
                assert(FitMod.fun=="simRL_2alpha" && contains(SimMod.fun,"RDMP"));
                SimMod.label  = SimMod.fun;
                SimMod.params = SimMod.simpar; 
                FitMod.label  = FitMod.fun;
                if fit_beta_flag
                    fitfunc_handle = str2func("fun"+FitMod.name); % if fitting beta together
                    lb = [0   1 0]; 
                    ub = [1 100 1];
                else
                    fitfunc_handle = str2func("fun"+FitMod.name+"_fixedBeta"); % if enforcing the same beta
                    lb = [0  0]; 
                    ub = [1  1];
                end
                
                CompStat = struct;
                CompStat.eff_plus = []; 
                CompStat.eff_minus = [];
        
                % set up environment
                simEnv = SetUp_RewardEnvironment(RewProbs, ne);
        
                if sessfit_flag
                    % compile data for this 'session' (n blocks)
                    sess_dat = cell(1,numSim);
                    for ns = 1:numSim
                        tempStats = Simulate_ModelChoice_randR(SimMod, simEnv, 'AB', ns+(ne-1)*numSim);
                        sess_dat{ns} = {tempStats.c, tempStats.r, beta1};
                    end
                    % fit params
                    qpar = cell(numFit*2,1); NegLL = nan(numFit*2,1);
                    for ii = 1:numFit*2
                        initPars = rand(1,length(ub));
                        [qpar{ii}, NegLL(ii)] = obj.fit_mult_blocks(sess_dat, fitfunc_handle, initPars, lb, ub);
                    end
                    fpar = qpar(min(NegLL)==NegLL);
                end
        
                % loop through repeated blocks for this env.
                tempTrials = struct;
                for ns = 1:numSim
                    if ~sessfit_flag
                        % fit RL model for inidividual blocks
                        tempStats = Simulate_ModelChoice_randR(SimMod, simEnv, 'AB', ns+(ne-1)*numSim);
            
                        qpar = cell(numFit,1); NegLL = nan(numFit,1);
                        for ii = 1:numFit
                            initPars = rand(1,length(ub)); %disp(initPars);
                            [qpar{ii}, NegLL(ii)] = fmincon(fitfunc_handle, initPars, [], [], [], [], lb, ub, [], op, {tempStats.c, tempStats.r, beta1});
                        end
                        fpar = qpar(min(NegLL)==NegLL);  
                    end
        
                    CompStat.eff_plus = [CompStat.eff_plus; ones(1,blockL-1)*fpar{1}(1)];
                    CompStat.eff_minus = [CompStat.eff_minus; ones(1,blockL-1)*fpar{1}(2)];
        
                    % use fitted RL to simulate the data
                    FitMod.params = [fpar{1}(1), beta1, fpar{1}(2)];
                    simStats = Simulate_ModelChoice_randR(FitMod, simEnv, 'AB', ns+(ne-1)*numSim);
        
                    stay  = simStats.c(1:end-1)==simStats.c(2:end);
                    prevR = simStats.r(1:end-1); 
                    prevO = simStats.c(1:end-1)==simEnv.hr_stim(1:end-1);
                    % prevRO = binary_to_decimal([prevR, prevO]);
        
                    % compute moving-window metrics through trials
                    for t = windowL:blockL-1
                        trial_idx = t-windowL+1:t;
                        tempTrials = obj.compute_and_append_metrics(tempTrials, {stay(trial_idx), prevR(trial_idx), prevO(trial_idx)});
                    end
                    Fi = fieldnames(tempTrials);
                    for f = 1:length(Fi)
                        EntTrial(ne).(Fi{f}) = tempTrials.(Fi{f}); % 1-by-trials MAT
                    end 
        
                end
        
                % store effective learn. rates
                EffAlphas(ne).eff_plus = mean(CompStat.eff_plus, 1,'omitnan');
                EffAlphas(ne).eff_minus = mean(CompStat.eff_minus, 1,'omitnan');
            end
        
            disp("Fit & sim complete: "+num2str(toc(fStart)/60,3)+" min elapsed.");
        
            % reshape output data
            Fi = fieldnames(EntTrial);
            for f = 1:length(Fi)
                MovMetOut{end}.(Fi{f}) = reshape([EntTrial.(Fi{f})], numTrials, [])'; % 1-by-trials MAT
            end
            MovMetOut{end}.eff_plus = reshape([EffAlphas.eff_plus], [], numEnv)';
            MovMetOut{end}.eff_minus = reshape([EffAlphas.eff_minus], [], numEnv)';
        
            %% save output file
            save("output/"+fname+".mat", 'MovMetOut','models');
            disp("File saved: "+ fname);
            disp(datetime);
        end

        %% Trial version
        function [TrialMetOut, models] = obtain_Trial_metrics_RDMP(obj, params)
            % simulate metrics from three models: RL1, RDMP, and RL2 (fitted to RDMP)
            % first simulate the first two models
            % then fit the third RL2 model to RDMP and simulate

            RewProbs = obj.config.RewProbs;
            blockL = size(RewProbs, 1);
            numTrials = blockL - 1;

            numSim = obj.config.numSim;
            numEnv = obj.config.numEnv;
            sessfit_flag = obj.config.sessfit_flag;
        
            %% Initialize models    
            alph0 = params(1);
            beta1 = params(2);
            mp1   = params(3);
            models = obj.init_sim_models([alph0, beta1, mp1, alph0]);
        
            fname = "TrialMet_"+(length(models)+1)+"mods_a"+alph0+"_b"+beta1+"_m"+mp1+"_n"+numEnv+"x"+numSim;
            if sessfit_flag
                fname = fname + "_sessFit";
            end
            
            if exist(fname+".mat",'file')  
                load("output/"+fname+".mat", 'TrialMetOut','models');
                disp("File exists, loading simulation output: "+fname);
                return;
            end
           
            %% Simulate moving window metrics from first two models
            TrialMetOut = cell(1, length(models)+1);
            for m = 1:length(models)
                player = struct;
                player.label = models{m}.fun;
                player.params = models{m}.simpar; 
                
                clear EntTrial EffAlphas
                EntTrial(numEnv) = struct; % across trials w.r.t. rev 
                EffAlphas(numEnv) = struct;
                %%
                parfor ne = 1:numEnv              
                    CompStat = struct;        
                    CompStat.stay  = [];
                    CompStat.prevR = [];
                    CompStat.prevO = [];
                    CompStat.eff_plus = []; 
                    CompStat.eff_minus = [];

                    simEnv = SetUp_RewardEnvironment(RewProbs, ne);
                    %% loop through repeated blocks for this env.  
                    for ns = 1:numSim
                        simStats = Simulate_ModelChoice_randR(player, simEnv, 'AB', ns+(ne-1)*numSim);                
                        stay  = simStats.c(1:end-1)==simStats.c(2:end);
                        prevR = simStats.r(1:end-1); 
                        prevO = simStats.c(1:end-1)==simEnv.hr_stim(1:end-1);                
                        
                        CompStat.stay =  [CompStat.stay; stay'];
                        CompStat.prevR = [CompStat.prevR; prevR'];
                        CompStat.prevO = [CompStat.prevO; prevO'];
        
                        if contains(player.label,"RDMP")
                            Q = [simStats.q1, simStats.q2];
                            Q_c = Q(sub2ind(size(Q), (1:blockL)', (simStats.c/2)+1.5)); % Q_chosen
                            DeltaQ = [diff(Q,[],1); nan(1,2)];
                            DeltaQc = DeltaQ(sub2ind(size(Q), (1:blockL)', (simStats.c/2)+1.5)); % deltaQ_chosen                
                            eff_alpha = DeltaQc ./ (simStats.r - Q_c); % effective learning rate    
                            eff_plus  = eff_alpha; eff_plus(simStats.r==0) = NaN;
                            eff_minus = eff_alpha; eff_minus(simStats.r==1) = NaN;
                            CompStat.eff_plus = [CompStat.eff_plus; eff_plus'];
                            CompStat.eff_minus = [CompStat.eff_minus; eff_minus'];
                        end                
                    end
        
                    % calculate metric for each trial position
                    tempTrials = struct;
                    for t = 1:blockL-1
                        stay  = CompStat.stay(:,t);
                        prevR = CompStat.prevR(:,t);
                        prevO = CompStat.prevO(:,t);
                        % prevRO = binary_to_decimal([prevR, prevO]);
                        tempTrials = obj.compute_and_append_metrics(tempTrials, {stay, prevR, prevO});
                    end
        
                    Fi = fieldnames(tempTrials);
                    for f = 1:length(Fi)
                        EntTrial(ne).(Fi{f}) = tempTrials.(Fi{f}); % 1-by-trials MAT
                    end
                    
                    % store effective learn. rates
                    if contains(player.label,"RDMP")
                        EffAlphas(ne).eff_plus  = mean(CompStat.eff_plus , 1,'omitnan');
                        EffAlphas(ne).eff_minus = mean(CompStat.eff_minus, 1,'omitnan');
                    end
                end
                % reshape output data        
                Fi = fieldnames(EntTrial);
                for f = 1:length(Fi)
                    TrialMetOut{m}.(Fi{f}) = reshape([EntTrial.(Fi{f})], numTrials,[])'; % N-by-trials MAT
                end
        
                if contains(player.label,"RDMP")
                    TrialMetOut{m}.eff_plus  = reshape([EffAlphas.eff_plus], [], numEnv)';
                    TrialMetOut{m}.eff_minus = reshape([EffAlphas.eff_minus], [], numEnv)';
                end
            end
            
            %% Fit RL2 to choice behavior of RDMP, and obtain moving window metrics
            fStart = tic;
            fit_beta_flag = 0;
        
            % optimize options
            numFit = 5;
            op = optimset('fminsearch');
            op.MaxIter = 1e7; op.MaxFunEvals = 1e7;
           
            clear EntTrial EffAlphas
            EntTrial(numEnv) = struct; % across trials w.r.t. rev 
            EffAlphas(numEnv) = struct;
            parfor ne = 1:numEnv
                fpar = [];
                FitMod = models{1}; % RL2
                SimMod = models{2}; % RDMP
                assert(FitMod.fun=="simRL_2alpha" && contains(SimMod.fun,"RDMP"));
                SimMod.label  = SimMod.fun;
                SimMod.params = SimMod.simpar; 
                FitMod.label  = FitMod.fun;
                if fit_beta_flag
                    fitfunc_handle = str2func("fun"+FitMod.name); % if fitting beta together
                    lb = [0   1 0]; 
                    ub = [1 100 1];
                else
                    fitfunc_handle = str2func("fun"+FitMod.name+"_fixedBeta"); % if enforcing the same beta
                    lb = [0  0]; 
                    ub = [1  1];
                end
                CompStat = struct;
                CompStat.stay  = [];
                CompStat.prevR = [];
                CompStat.prevO = [];
                CompStat.eff_plus = []; 
                CompStat.eff_minus = [];
        
                % set up environment
                simEnv = SetUp_RewardEnvironment(RewProbs, ne);
        
                if sessfit_flag
                    % compile data for this 'session' (n blocks)
                    sess_dat = cell(1,numSim);
                    for ns = 1:numSim
                        tempStats = Simulate_ModelChoice_randR(SimMod, simEnv, 'AB', ns+(ne-1)*numSim);
                        sess_dat{ns} = {tempStats.c, tempStats.r, beta1};
                    end
                    % fit params
                    qpar = cell(numFit*2,1); NegLL = nan(numFit*2,1);
                    for ii = 1:numFit*2
                        initPars = rand(1,length(ub));
                        [qpar{ii}, NegLL(ii)] = obj.fit_mult_blocks(sess_dat, fitfunc_handle, initPars, lb, ub);
                    end
                    fpar = qpar(min(NegLL)==NegLL);        
                end
        
                % loop through repeated blocks for this env.
                for ns = 1:numSim
                    if ~sessfit_flag
                        % fit RL model for inidividual blocks
                        tempStats = Simulate_ModelChoice_randR(SimMod, simEnv, 'AB', ns+(ne-1)*numSim);
                        qpar = cell(numFit,1); NegLL = nan(numFit,1);
                        for ii = 1:numFit
                            initPars = rand(1,length(ub)); %disp(initPars);
                            [qpar{ii}, NegLL(ii)] = fmincon(fitfunc_handle, initPars, [], [], [], [], lb, ub, [], op, {tempStats.c, tempStats.r, beta1});
                        end
                        fpar = qpar(min(NegLL)==NegLL);  
                    end            
        
                    % use fitted RL to simulate the data
                    FitMod.params = [fpar{1}(1), beta1, fpar{1}(2)];
                    simStats = Simulate_ModelChoice_randR(FitMod, simEnv, 'AB', ns+(ne-1)*numSim);
        
                    stay  = simStats.c(1:end-1)==simStats.c(2:end);
                    prevR = simStats.r(1:end-1); 
                    prevO = simStats.c(1:end-1)==simEnv.hr_stim(1:end-1);
        
                    CompStat.stay =  [CompStat.stay; stay'];
                    CompStat.prevR = [CompStat.prevR; prevR'];
                    CompStat.prevO = [CompStat.prevO; prevO'];
                    CompStat.eff_plus  = [CompStat.eff_plus; ones(1,blockL-1)*fpar{1}(1)];
                    CompStat.eff_minus = [CompStat.eff_minus; ones(1,blockL-1)*fpar{1}(2)];                     
                end
        
                % calculate metric for each trial position
                tempTrials = struct;
                for t = 1:blockL-1
                    stay  = CompStat.stay(:,t);
                    prevR = CompStat.prevR(:,t);
                    prevO = CompStat.prevO(:,t);
                    % prevRO = binary_to_decimal([prevR, prevO]);
                    tempTrials = obj.compute_and_append_metrics(tempTrials, {stay, prevR, prevO});
                end
                % assign metric output 
                Fi = fieldnames(tempTrials);
                for f = 1:length(Fi)
                    EntTrial(ne).(Fi{f}) = tempTrials.(Fi{f}); % 1-by-trials MAT
                end 
        
                % store effective learn. rates
                EffAlphas(ne).eff_plus = mean(CompStat.eff_plus, 1,'omitnan');
                EffAlphas(ne).eff_minus = mean(CompStat.eff_minus, 1,'omitnan');
            end
        
            disp("Fit & sim complete: "+num2str(toc(fStart)/60,3)+" min elapsed.");
        
            % reshape output data
            Fi = fieldnames(EntTrial);
            for f = 1:length(Fi)
                TrialMetOut{end}.(Fi{f}) = reshape([EntTrial.(Fi{f})], numTrials, [])'; % 1-by-trials MAT
            end
            TrialMetOut{end}.eff_plus = reshape([EffAlphas.eff_plus], [], numEnv)';
            TrialMetOut{end}.eff_minus = reshape([EffAlphas.eff_minus], [], numEnv)';
        
            %% save output file
            save("output/"+fname+".mat", 'TrialMetOut','models');
            disp("File saved: "+ fname);
            disp(datetime);
        end

        %% testing decoder on simulated model metrics
        function [PostProb, AccuMAT] = test_decoder_MovWin_metrics(obj, ModOutput)
            % Test decoding accuracies on three models: RL1, RDMP, and RL2 (fitted to RDMP)   
            if contains(obj.config.metric_type,"diff")
                diff_flag = 1;
            else
                diff_flag = 0;
            end
            blockL = size(obj.config.RewProbs,1);
            numBlocks = obj.config.numSim * obj.config.numEnv;
            assert(size(ModOutput{1}.ERDS,1)==numBlocks);
        
            numTrials = size(ModOutput{1}.ERDS,2);
            disp("Confirm # of trials: "+numTrials);
            assert(numel(obj.config.windowL:blockL-1)==numTrials);
        
            PostProb = cell(length(obj.config.VarSet), length(ModOutput));
            AccuMAT  = PostProb;
        
            % loop through each model
            for m = 1:length(ModOutput)
        
                % loop through each set of features used for decoding
                for f_set = 1:length(obj.config.VarSet)
                    this_vars = obj.config.VarSet{f_set};
                    if strcmp(this_vars,"all")
                        this_vars = string(fieldnames(ModOutput{1})); % first cell array to keep out eff. alpha's
                    end
        
                    % loop through each trial data to obatin posterior prob
                    prob_optim = nan(obj.config.numEnv, numTrials);
                    accu_optim = nan(obj.config.numEnv, numTrials);
                    for tt = 1:numTrials
                        T_dat = struct;
                        for ii = 1:length(this_vars)
                            tempDat = ModOutput{m}.(this_vars(ii))(:,tt);
                            if diff_flag
                                tempDat = tempDat - ModOutput{m}.(this_vars(ii))(:,1);
                            end
                            T_dat.(this_vars(ii)) = mean(reshape(tempDat, [], obj.config.numSim),2,'omitnan');
                            % T_dat.(this_vars(ii)) = tempDat;
                        end               
                       
                        T_dat = struct2table(T_dat);
                        if obj.config.conv_nan2zero
                            % convert NaN into zero?
                            numVars = vartype('numeric');
                            T_dat(:, numVars) = fillmissing(T_dat(:, numVars), 'constant', 0);
                        end
        
                        [labels, scores] = predict(obj.TrainedDecoder{f_set,tt}, T_dat);
                        prob_optim(:,tt) = scores(:,2);
        
                        % obtain true labels (effective alpha's)
                        if ~isfield(ModOutput{m},'eff_plus')
                            true_label = zeros(size(labels)); % DeltaAlpha = 0 for RL1
                        else
                            Delta_alpha = ModOutput{m}.eff_plus - ModOutput{m}.eff_minus;
                            trial_idx = tt:tt+obj.config.windowL-1;
                            true_label = mean(Delta_alpha(:,trial_idx),2,'omitnan') > 0;
                        end
                        accu_optim(:,tt) = labels==true_label;
                    end
        
                    PostProb{f_set, m} = prob_optim;
                    AccuMAT{f_set, m} = accu_optim;
                end
            end
            disp("Decoding complete");
        end

        %% Trial version
        function [PostProbMAT, AccuMAT] = test_decoder_Trial_metrics(obj, ModOutput)
        % Test decoding accuracies on three models: RL1, RDMP, and RL2 (fitted to RDMP)
            DecodeVar_set = obj.config.VarSet;
            blockL = size(obj.config.RewProbs,1);
            numEnv = obj.config.numEnv;
            % numSim = obj.config.numSim;
            assert(size(ModOutput{1}.ERDS,1)==numEnv);
        
            numTrials = size(ModOutput{1}.ERDS,2);
            disp("Confirm # of trials: "+numTrials);
            assert(blockL-1==numTrials);
        
            PostProbMAT = cell(length(DecodeVar_set), length(ModOutput));
            AccuMAT = cell(length(DecodeVar_set), length(ModOutput));
        
            for m = 1:length(ModOutput)
                for f_set = 1:length(DecodeVar_set)
                    % disp(f_set);
                    this_vars = DecodeVar_set{f_set};
                    if strcmp(this_vars,"all")
                        this_vars = string(fieldnames(ModOutput{1})); % first cell array to keep out eff. alpha's
                    end
        
                    % loop through each trial data to obatin posterior prob
                    % prob_optim = nan(numEnv, numTrials);
                    % accu_optim = nan(numEnv, numTrials);
                    for tt = 1:numTrials
                        T_dat = struct;
                        for ii = 1:length(this_vars)
                            tempDat = ModOutput{m}.(this_vars(ii))(:,tt);
                            % T_dat.(this_vars(ii)) = mean(reshape(tempDat, [], numEnv), 2, 'omitnan');
                            T_dat.(this_vars(ii)) = tempDat;
                        end               
                       
                        T_dat = struct2table(T_dat);
                        if obj.config.conv_nan2zero
                            % convert NaN into zero?
                            numVars = vartype('numeric');
                            T_dat(:, numVars) = fillmissing(T_dat(:, numVars), 'constant', 0);
                        end
        
                        [labels, scores] = predict(obj.TrainedDecoder{f_set,tt}, T_dat);
                        prob_optim(:,tt) = scores(:,2);
        
                        % obtain true labels (effective alpha's)
                        if ~isfield(ModOutput{m},'eff_plus')
                            true_label = zeros(size(labels)); % DeltaAlpha = 0 for RL1
                        else
                            Delta_alpha = ModOutput{m}.eff_plus - ModOutput{m}.eff_minus;
                            true_label = mean(Delta_alpha(:,tt),1,'omitnan') > 0;
                        end
                        accu_optim(:,tt) = labels==true_label;
                    end
        
                    PostProbMAT{f_set, m} = prob_optim;
                    AccuMAT{f_set, m} = accu_optim;
                end
            end
            disp("Decoding complete");
        end

        %% Cross validate functions for given data
        function [PostProb, AccuMAT] = crossval_Trial_metrics(obj, CrossDat)
            DecodeVar_set = obj.config.VarSet;
            blockL = size(obj.config.RewProbs,1);
            numEnv = obj.config.numEnv;
            assert(size(CrossDat{1}.ERDS,1)==numEnv);

            numTrials = size(CrossDat{1}.ERDS,2);
            disp("Confirm # of trials: "+numTrials);
            % assert(blockL-1==numTrials);
        
            PostProb = cell(length(DecodeVar_set), 1);
            AccuMAT = cell(length(DecodeVar_set), 1);
            tic
            for f_set = 1:length(DecodeVar_set)
                this_vars = DecodeVar_set{f_set};
                if strcmp(this_vars,"all")
                    this_vars = string(fieldnames(CrossDat{1}));
                    this_vars = setdiff(this_vars, ["eff_plus","eff_minus"]);
                end

                % loop through each trial point to perform decoding
                for tt = 1:numTrials
                    T_dat = struct;
                    for ci = 1:length(CrossDat)
                        tempMod = struct;
                        for ii = 1:length(this_vars)
                            tempDat = CrossDat{ci}.(this_vars(ii))(:,tt);
                            tempMod.(this_vars(ii)) = tempDat;
                        end 
                        tempMod.model_label = repmat(ci==1, size(tempMod.(this_vars(ii)),1), 1);
                        T_dat = append_to_fields(T_dat, {tempMod});
                    end                       

                    T_dat = struct2table(T_dat);
                    if obj.config.conv_nan2zero
                        % convert NaN into zero?
                        numVars = vartype('numeric');
                        T_dat(:, numVars) = fillmissing(T_dat(:, numVars), 'constant', 0);
                    end

                    cvMdl = fitclinear(T_dat, 'model_label','KFold',10,'Learner','logistic');

                    prob_optim(:,tt) = 1 - kfoldLoss(cvMdl,'LossFun','logit','Mode','individual');
                    accu_optim(:,tt) = 1 - kfoldLoss(cvMdl,'LossFun','classiferror','Mode','individual');
                end

                PostProb{f_set, 1} = prob_optim;
                AccuMAT{f_set, 1} = accu_optim;

            end
            disp("Decoding complete");
            toc
        end

        function [PostProb, AccuMAT] = crossval_MovWin_metrics(obj, CrossDat)
            DecodeVar_set = obj.config.VarSet;
            % blockL = size(obj.config.RewProbs,1);
            numEnv = obj.config.numEnv;
            numSim = obj.config.numSim;
            assert(size(CrossDat{1}.ERDS,1)==numEnv*numSim);

            numTrials = size(CrossDat{1}.ERDS,2);
            disp("Confirm # of trials: "+numTrials);
            % assert(blockL-1==numTrials);
        
            PostProb = cell(length(DecodeVar_set), 1);
            AccuMAT = cell(length(DecodeVar_set), 1);
            tic
            for f_set = 1:length(DecodeVar_set)
                this_vars = DecodeVar_set{f_set};
                if strcmp(this_vars,"all")
                    this_vars = string(fieldnames(CrossDat{1}));
                    this_vars = setdiff(this_vars, ["eff_plus","eff_minus"]);
                end

                % loop through each trial point to perform decoding
                for tt = 1:numTrials
                    T_dat = struct;
                    for ci = 1:length(CrossDat)
                        tempMod = struct;
                        for ii = 1:length(this_vars)
                            tempDat = CrossDat{ci}.(this_vars(ii))(:,tt);
                            tempMod.(this_vars(ii)) = tempDat;
                        end 
                        tempMod.model_label = repmat(ci==1, size(tempMod.(this_vars(ii)),1), 1); % 1=RDMP, 0=RL2
                        T_dat = append_to_fields(T_dat, {tempMod});
                    end                       

                    T_dat = struct2table(T_dat);
                    if obj.config.conv_nan2zero
                        % convert NaN into zero?
                        numVars = vartype('numeric');
                        T_dat(:, numVars) = fillmissing(T_dat(:, numVars), 'constant', 0);
                    end

                    cvMdl = fitclinear(T_dat, 'model_label','KFold',10,'Learner','logistic');

                    prob_optim(:,tt) = 1 - kfoldLoss(cvMdl,'LossFun','logit','Mode','individual');
                    accu_optim(:,tt) = 1 - kfoldLoss(cvMdl,'LossFun','classiferror','Mode','individual');
                end

                PostProb{f_set, 1} = prob_optim;
                AccuMAT{f_set, 1} = accu_optim;

            end
            disp("Decoding complete");
            toc


        end
    end

    %% static methods
    methods (Static)        
        function DecompMap = intialize_decomp_map()
            DecompMap = struct;

            DecompMap.R = containers.Map({0,1},{'lose','win'});
            DecompMap.O = containers.Map({0,1},{'worse','better'});
            DecompMap.RO = containers.Map({0,1,2,3},{'loseworse','losebetter','winworse','winbetter'});
        end
        
        %% entropy calculation function
        function tempTrials = compute_and_append_metrics(tempTrials, input_dat)
            R_decomp = containers.Map({0,1},{'lose','win'});
            O_decomp = containers.Map({0,1},{'worse','better'});
            RO_decomp = containers.Map({0,1,2,3},{'loseworse','losebetter','winworse','winbetter'});

            stay  = input_dat{1};
            prevR = input_dat{2};
            prevO = input_dat{3};
            prevRO = binary_to_decimal([prevR, prevO]);            

            tempMet = struct;
            tempMet.H_str = Shannon_Entropy(stay);
            tempMet = append_to_fields(tempMet, {Conditional_Entropy_decomp(stay, prevR, "ERDS", R_decomp), ...
                                Mutual_Information_decomp(stay, prevR, "MIRS", R_decomp), ...
                                Conditional_Entropy_decomp(stay, prevO, "EODS", O_decomp ), ...
                                Mutual_Information_decomp(stay, prevO, "MIOS", O_decomp), ...
                                Conditional_Entropy_decomp(stay, prevRO, "ERODS", RO_decomp), ...
                                Mutual_Information_decomp(stay, prevRO, "MIROS", RO_decomp) });
            tempTrials = append_to_fields(tempTrials, {tempMet});
        end

        %% misc.: for initializing/simulation/fitting models
        function [models] = init_sim_models(params, idx)
            models = {};    % initialize as cell array
            alph1 = params(1);
            beta1 = params(2);
            mp1   = params(3);
            alph2 = params(4);
        
            % find base alpha_1 for RDMP models which gives the same intial effective learning rate
            N_meta  = 4;         % # of meta-states
            f = @(x) mean(x .^ ( ( (N_meta-2)*(1:N_meta) + 1 ) / (N_meta-1))) - alph1;    
            options = optimoptions('fsolve', 'Display', 'off'); % 'iter' to show steps; use 'off' to suppress
            RDMP_a1 = fsolve(f, alph1, options);
            disp("RL learning rate: \alpha = "+alph1);
            disp("Corresponding RDMP q_1 = "+RDMP_a1);
        
            % 1. RL2
            m = length(models) + 1;
            models{m}.name = 'RL_2alpha';      
            models{m}.fun = 'simRL_2alpha'; % specify simulation function   
            models{m}.simpar  =[alph1, beta1, alph2];       
            models{m}.lb      =[ 0   1  0];   
            models{m}.ub      =[ 1 100  1];   
            models{m}.label = "RL";    
            models{m}.plabels = ["\alpha+", "\beta", "\alpha-"];
        
            % 2. RDMP
            m = length(models) + 1;
            models{m}.name = 'RDMP_step';      
            models{m}.fun = 'simRDMP_step'; 
            models{m}.simpar  =[RDMP_a1, beta1, mp1];      
            models{m}.lb      =[ 0   1  0]; 
            models{m}.ub      =[ 1 100  1]; 
            models{m}.label = "RDMP_{two-set}";    
            models{m}.plabels = ["\alpha_1", "\beta", "m_1"];
        
            if ~exist('idx','var')
                idx = 1:length(models);
            end
            models = models(idx);
            disp(length(models)+" models initialized");
        end

        % fitting func (session-fit)
        function [qpar, negLL] = fit_mult_blocks(sess_dat, fitfunc_handle, initPars, lb, ub)
            op = optimset('fminsearch');
            op.MaxIter = 1e7; op.MaxFunEvals = 1e7;
        
            % func_handle = str2func("block_wrapper");
            func_handle = @RDMP_Decoder_class.block_wrapper;
            [qpar, negLL, exitflag] = fmincon(func_handle, initPars, [], [], [], [], lb, ub, [], op, {sess_dat, fitfunc_handle});
        
            if exitflag==0
                % disp("Didn't converge");
                qpar = nan(size(qpar));   %did not converge to a solution, so return NaN
                negLL = nan;
            end
        end

        function [tot_negloglike, nLL] = block_wrapper(xpar, input_dat)
            tot_negloglike = 0;
            fit_dat = input_dat{1};
            fit_handle = input_dat{2};
            numSess = length(fit_dat);
        
            for i = 1:numSess
                stats = fit_dat{i};
                [this_LL] = fit_handle(xpar, stats);
                
                tot_negloglike = tot_negloglike + this_LL;
                nLL(i) = this_LL;  
            end
        end
    end    
end
