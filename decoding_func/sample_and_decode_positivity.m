function [PosNegAccu, AllGroupDat] = sample_and_decode_positivity(RewProbs, DecodeOpt, opt_alphas, beta1)
    display_on = 0;    % display results

    DecodeVar_set = DecodeOpt.VarSet;   
    windowL = DecodeOpt.windowL;
    agentN  = DecodeOpt.agentN;
    sampleN = DecodeOpt.sampleN;
    withinN = DecodeOpt.withinN;
    radi    = DecodeOpt.radius;
    sigma   = DecodeOpt.sigma;
    trial_port = DecodeOpt.trial_portion;

    k_fold = 10;       % K-fold CV results: 1)logistic, 2)SVM, 3)log+rbf, 4)SVM+rbf 
    conv_nan2zero = 1; % convert NaN's into zero?
    

    % intialize model
    SimMod = struct;
    SimMod.name = 'RL_2alpha';      
    SimMod.fun = 'simRL_2alpha'; % specify simulation function   
    SimMod.initpar =[.5   5 .5];       
    SimMod.lb      =[ 0   1  0];   
    SimMod.ub      =[ 1 100  1];   
    SimMod.label = "RL2";    
    SimMod.plabels = ["\alpha_+", "\beta", "\alpha_-"];

    R_decomp = containers.Map({0,1},{'lose','win'});
    O_decomp = containers.Map({0,1},{'worse','better'});
    RO_decomp = containers.Map({0,1,2,3},{'loseworse','losebetter','winworse','winbetter'});
    
    disp("   > Decoding from: "+trial_port+" "+windowL+" trials");
    blockL = size(RewProbs,1);
    simStart = tic;
    AllGroupDat = struct;
    for group = ["optimistic","pessimistic"]
        disp(" >> Simulating: "+group+" group");
        clear agentMet 
        agentMet(agentN) = struct;
        for n = 1:agentN
            % disp(n);    
            %% sample agent params for each block 
            alpha_plus = []; alpha_minus = []; % clear par vars
            switch group
                case "optimistic"
                    alpha_plus  = opt_alphas(n,2); % note this is y-coord in the grid
                    alpha_minus = opt_alphas(n,1); assert(alpha_plus>alpha_minus);
                case "pessimistic"
                    alpha_plus  = opt_alphas(n,1); % flipped from optimistic case
                    alpha_minus = opt_alphas(n,2); assert(alpha_plus<alpha_minus);
            end
    
            % sample parameters (alpha's) for this agent
            x_y = [];
            while length(x_y)<sampleN
                N_batch = ceil((sampleN - length(x_y)) * 1.5);
                samples = mvnrnd([alpha_minus, alpha_plus], sigma^2 * eye(2), N_batch); % input as [x, y] coord
                dists = sqrt((samples(:,1) - alpha_minus).^2 + (samples(:,2) - alpha_plus).^2);        
                x_y = [x_y; samples(dists <= radi,:)];
            end
            x_y = x_y(1:sampleN,:);
            x_y = max(0, min(x_y,1)); % cap alpha between [0 1]

            %% generate behavior w/ each sample of (alpha-, alpha+); just once
            % agentMet(n) = struct;
            agentMet(n).label_group = repmat(group, 1, sampleN);
            agentMet(n).agent_id = repmat(n, 1, sampleN);

            DeltaAlpha = x_y(:,2) - x_y(:,1);
            agentMet(n).DeltaAlpha = DeltaAlpha';

            % set up environment and agent
            simEnv = SetUp_RewardEnvironment(RewProbs, n);                        
            player = struct;
            player.label = SimMod.fun;
            for k = 1:sampleN               
                player.params = [x_y(k,2), beta1, x_y(k,1)];  % alpha+, beta, alpha-             
    
                % within-sample averages?
                tempAvg = struct;
                for nn = 1:withinN
                    simStats = Simulate_ModelChoice_randR(player, simEnv, 'AB', k+(n-1)*sampleN+nn); 
                    stay  = simStats.c(1:end-1)==simStats.c(2:end);
                    prevStay = [NaN; stay(1:end-1)]; % take 1-step history of Y
                    prevR = simStats.r(1:end-1);
                    prevO = simStats.c(1:end-1)==simEnv.hr_stim(1:end-1);                    
                    
                    switch trial_port 
                        case "last"
                            trial_idx = (1+blockL-windowL:blockL)-1;
                        case "first"
                            trial_idx = 1:windowL;
                    end
                    % take relevant portion of the block only
                    stay = stay(trial_idx);
                    prevStay = prevStay(trial_idx);
                    prevR = prevR(trial_idx);
                    prevO = prevO(trial_idx);
                    prevRO = binary_to_decimal([prevR, prevO]); 
    
                    tempAvg.H_str = Shannon_Entropy(stay);
                    tempAvg = append_to_fields(tempAvg, {Conditional_Entropy_weighted(stay, prevR, "ERDS", R_decomp), ...
                                                Mutual_Information_weighted(stay, prevR, "MIRS", R_decomp), ...
                                                Conditional_Mutual_Information(prevR, stay, prevStay, "TERS", R_decomp), ...
                                                Conditional_Entropy_weighted(stay, prevO, "EODS", O_decomp ), ...
                                                Conditional_Mutual_Information(prevO, stay, prevStay, "TEOS", O_decomp), ...
                                                Mutual_Information_weighted(stay, prevO, "MIOS", O_decomp), ...
                                                Conditional_Entropy_weighted(stay, prevRO, "ERODS", RO_decomp), ...
                                                Mutual_Information_weighted(stay, prevRO, "MIROS", RO_decomp), ...
                                                Conditional_Mutual_Information(prevRO, stay, prevStay, "TEROS", RO_decomp) });
                end
                Fi = fieldnames(tempAvg);
                for f = 1:length(Fi)
                    agentMet(n).(Fi{f})(k) = mean(tempAvg.(Fi{f}),'omitnan');
                end  
            end          
        end
        %% store each group data
        tempGroup = struct;
        Fi = fieldnames(agentMet);
        for f = 1:length(Fi)
            tempGroup.(Fi{f}) = [agentMet.(Fi{f})]';
        end
        AllGroupDat = append_to_fields(AllGroupDat, {tempGroup});
    end
    simEnd = toc(simStart);
    disp("Sim complete: "+simEnd+ "sec");
    
    %% Run decoding from obtained samples & labels 
    decStart = tic;

    T = struct2table(AllGroupDat);
    if conv_nan2zero
        % convert NaN into zero?
        numVars = vartype('numeric');
        T(:, numVars) = fillmissing(T(:, numVars), 'constant', 0);
    end
    T_clean = T(~any(ismissing(T), 2), :);
    disp("    Data length = "+height(T));
    % disp("    ~NaN length = "+height(T_clean));      
    
    %% Loop through each set of features and obtain accuracy for each
    PosNegAccu = struct;
    for f_set = 1:length(DecodeVar_set)
        fields_to_keep = ["label_group", "agent_id", DecodeVar_set{f_set}];            
        
        T_reduced = table;
        for ii = 1:length(fields_to_keep)
            T_reduced.(fields_to_keep(ii)) = T_clean.(fields_to_keep(ii));
        end

        %% Test Optimistic vs. Pessimistic
        test_idx = T_reduced.label_group=="optimistic"|T_reduced.label_group=="pessimistic";
        T_test = T_reduced(test_idx,:);
        T_test.label_group = (T_test.label_group=="optimistic");
        
        % K-fold CV
        [PosNegAccu.Kfold{f_set}] = test_decoder_Kfold(T_test, k_fold);   
        
        % (Leave-one-out CV) Optimistic vs. Neutral
        [PosNegAccu.LOO{f_set}] = test_decoder_LeaveOneAgentOut(T_test, agentN);
    
        if display_on
            disp(f_set+": Optimisitc vs. Pessimistic?");
            disp("     >>> Mean accu (K-fold) = "+strjoin(string(PosNegAccu.Kfold{f_set}),", "));
            disp("     >>> (Leave-one-agent-out) = "+mean(PosNegAccu.LOO{f_set}));
        end
    end    
    disp("Decoding complete: "+toc(decStart)/60+" min");
end

%% subfunc: K-fold cross validation
function [avg_accu] = test_decoder_Kfold(T_test, k_fold)
    avg_accu = nan(k_fold, 4);

    % must remove non-predictor variables in the table before decoding
    vars_to_remove = {'agent_id', 'DeltaAlpha'};
    for v = 1:length(vars_to_remove)
        if ismember(vars_to_remove{v}, T_test.Properties.VariableNames)
            T_test = removevars(T_test, vars_to_remove{v});
        end
    end

    % down-sample K times for a balanced training/test set
    for k = 1:k_fold       
        cat_num(1) = sum(T_test.label_group==true);
        cat_num(2) = sum(T_test.label_group==false);
        numDrop = abs(diff(cat_num));

        rng(k);
        if cat_num(1) > cat_num(2)
            cat_idx = find(T_test.label_group==true);
            drop_idx = cat_idx(randperm(cat_num(1), numDrop));
        else
            cat_idx = find(T_test.label_group==false);
            drop_idx = cat_idx(randperm(cat_num(2), numDrop));
        end
        keep_idx = true(height(T_test),1);
        keep_idx(drop_idx) = false;
        T_down = T_test(keep_idx,:); % equalized sample       
        
        CV_mdl = fitclinear(T_down, 'label_group', 'KFold',k_fold, 'Learner','logistic');
        avg_accu(k,1) = 1 - kfoldLoss(CV_mdl, 'Mode','average', 'LossFun', 'classiferror'); % averaged loss 
        % Log_avgLogLoss(k) = kfoldLoss(CV_mdl, 'Mode','average', 'LossFun', 'logit');
    
        CV_mdl = fitclinear(T_down, 'label_group', 'KFold',k_fold, 'Learner','svm');
        avg_accu(k,2) = 1 - kfoldLoss(CV_mdl, 'Mode','average', 'LossFun', 'classiferror'); % averaged loss 
    
        CV_mdl = fitckernel(T_down, 'label_group', 'KFold',k_fold, 'Learner','logistic');
        avg_accu(k,3) = 1 - kfoldLoss(CV_mdl, 'Mode','average', 'LossFun', 'classiferror'); % averaged loss 
    
        CV_mdl = fitcsvm(T_down, 'label_group', 'KFold',k_fold, 'KernelFunction','rbf');
        avg_accu(k,4) = 1 - kfoldLoss(CV_mdl, 'Mode','average', 'LossFun', 'classiferror'); % averaged loss 
    end

    avg_accu = mean(avg_accu, 1);
end

%% subfunc: Leve-one-out procedure
function [avg_accu] = test_decoder_LeaveOneAgentOut(T_test, agentN)
    avg_accu = nan(agentN, 2);

    % must remove non-predictor variables in the table before decoding
    vars_to_remove = {'DeltaAlpha'};
    for v = 1:length(vars_to_remove)
        if ismember(vars_to_remove{v},T_test.Properties.VariableNames)
            T_test = removevars(T_test, vars_to_remove{v});
        end
    end

    %% loop through each agent (in each group) to test
    test_labels = unique(T_test.label_group);
    for group_idx = 1:length(test_labels)
        for agent_idx = 1:agentN
            this_idx = T_test.label_group==test_labels(group_idx) & T_test.agent_id== agent_idx;    
            train_data = T_test(~this_idx,:);
            test_data  = T_test(this_idx,:);

            % down-sample K=10 times for a balanced training set, then C.V.
            tempAgentAcc = nan(10,1);
            for k = 1:10
                cat_num(1) = sum(train_data.label_group==true);
                cat_num(2) = sum(train_data.label_group==false);
                numDrop = abs(diff(cat_num));

                rng(k);
                if cat_num(1) > cat_num(2)
                    cat_idx = find(train_data.label_group==true);
                    drop_idx = cat_idx(randperm(cat_num(1), numDrop));
                else
                    cat_idx = find(train_data.label_group==false);
                    drop_idx = cat_idx(randperm(cat_num(2), numDrop));
                end
                keep_idx = true(height(train_data),1);
                keep_idx(drop_idx) = false;
                T_down = train_data(keep_idx,:); % equalized training sample  

                % train decoder (logistic)
                CV_mdl = fitclinear(T_down, 'label_group', 'Learner','logistic');
                [labels, ~] = predict(CV_mdl, test_data);
                correct = labels==test_data.label_group;
                % post_prob = scores(:,unique(test_data.label_group)+1); 
                tempAgentAcc(k) = mean(correct);
            end
            avg_accu(agent_idx, group_idx) = mean(tempAgentAcc,'omitnan');
        end
    end
    % return unrwapped data
    avg_accu = avg_accu(:);
end

