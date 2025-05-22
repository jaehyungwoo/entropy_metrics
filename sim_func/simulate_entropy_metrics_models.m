%% simulate function
function [ModOutput] = simulate_entropy_metrics_models(models, RewProbs, numEnv, numSim, steadyL, rev_pos, afterRevL)    
    ModOutput = cell(1, length(models));  
    disp("Simulating "+ length(models) + " models");
    if isnan(rev_pos)
        inclue_rev_flag = false;
    else
        inclue_rev_flag = true;
        assert(exist('afterRevL','var'), 'after reversal window length missing');
    end
    
    decomp_on = 1; % include decompositions?

    blockL = size(RewProbs,1);
    %% loop through each model and simulate behavior
    allSt = tic;
    for m = 1:length(models)     
        ModOutput{m} = struct; % initialize
        switch class(models)
            case 'cell'
                this_model = models{m};
            case 'struct'
                this_model = models(m);
        end
        player = struct;
        player.label = "sim" + this_model.name;
        player.params = this_model.simpar; 
    
        % initialize variable
        clear EntTrial EntBlock_all EntBlock_early EntBlock_steady EntBlock_rev EntBlock_before   
        EntTrial(numEnv) = struct; % across trials w.r.t. rev    
        EntBlock_all(numEnv) = struct; % within blocks (all trials)
        EntBlock_early(numEnv) = struct; % within blocks (first N trials only)
        EntBlock_steady(numEnv) = struct; % wihitn blocks (steady state only)
        EntBlock_rev(numEnv) = struct; % wihitn blocks (after rev only)
        EntBlock_before(numEnv) = struct; % wihitn blocks (before rev only)

        tic
        % loop through each batch (use parfor here for parallel computing)
        for ne = 1:numEnv
            CompStat = struct; % trial data to be compiled (for each batch)
                CompStat.stay  = []; % stay(1) or switch(0)
                CompStat.currO = [];
                
                CompStat.prevR = []; % rewarded(1) or not(0)
                CompStat.prevO = []; % chose better(0) or not(0)                        
                CompStat.prevRO = [];

                CompStat.firstC = [];
                CompStat.currC = [];
                CompStat.hr_opt = [];
                CompStat.cBetter = [];
                CompStat.lastR = [];
    
            % repeat random simulation of each environment   
            tempE = struct;
            tempE.All = struct;
            tempE.Early = struct;
            tempE.Steady = struct;
            tempE.afterRev = struct;
            tempE.beforeRev = struct;
    
            % Set up unique environment for each batch
            % Note: only the left/right position differs if using random reward generation
            simEnv = SetUp_RewardEnvironment(RewProbs, ne); 
    
            for ns = 1:numSim           
                % simulate task with given seed:            
                % this randomizes choice and reward generation
                simStats = Simulate_ModelChoice_randR(player, simEnv, 'AB', ns+(ne-1)*numSim); 

                cBetter = (simStats.c==simEnv.hr_stim);  
                stay  = simStats.c(1:end-1)==simStats.c(2:end);
                prevR = simStats.r(1:end-1); 
                prevO = cBetter(1:end-1); 
    
                % concatenate same trial data (for across-trial data)        
                CompStat.stay =  [CompStat.stay; stay']; % [blocks x trials]
                CompStat.prevR = [CompStat.prevR; prevR'];
                CompStat.lastR = [CompStat.lastR; simStats.r(end)];
                CompStat.prevO = [CompStat.prevO; prevO'];                

                % current trial info: shifted by 1 
                CompStat.firstC = [CompStat.firstC; simStats.c(1)==simEnv.hr_stim(1)];
                CompStat.currC = [CompStat.currC; simStats.c(2:end)'];
                CompStat.hr_opt = [CompStat.hr_opt; simEnv.hr_stim(2:end)'];
                % CompStat.currO = [CompStat.currO; cBetter(2:end)'];
                CompStat.cBetter = [CompStat.cBetter; cBetter'];
    
                % compute whole-block metrics
                [tempE.All] = compute_behavioral_and_entropy_met(tempE.All, decomp_on, simStats.c, simStats.r, simEnv.hr_stim, stay, prevR, prevO);

                % early portion: first N trials 
                f_idx = 1:steadyL;
                [tempE.Early] = compute_behavioral_and_entropy_met(tempE.Early, decomp_on, simStats.c(1:steadyL+1), simStats.r(1:steadyL+1), simEnv.hr_stim(1:steadyL+1), stay(f_idx), prevR(f_idx), prevO(f_idx));
                
                % N trials before reversal
                s_idx = rev_pos-steadyL+1:rev_pos;
                [tempE.beforeRev] = compute_behavioral_and_entropy_met(tempE.beforeRev, decomp_on, simStats.c(s_idx+1), simStats.r(s_idx+1), simEnv.hr_stim(s_idx+1), stay(s_idx), prevR(s_idx), prevO(s_idx));

                % steady state: last N trials within each phase
                s_idx = [rev_pos-steadyL+1:rev_pos, blockL-steadyL+1:blockL]-1;
                s_idx = s_idx(~isnan(s_idx));
                [tempE.Steady] = compute_behavioral_and_entropy_met(tempE.Steady, decomp_on, simStats.c(s_idx+1), simStats.r(s_idx+1), simEnv.hr_stim(s_idx+1), stay(s_idx), prevR(s_idx), prevO(s_idx));

                % after reversal: N trials after reversals
                if inclue_rev_flag
                    r_idx = (rev_pos+1:rev_pos+afterRevL) - 1;
                    [tempE.afterRev] = compute_behavioral_and_entropy_met(tempE.afterRev, decomp_on, simStats.c(r_idx+1), simStats.r(r_idx+1), simEnv.hr_stim(r_idx+1), stay(r_idx), prevR(r_idx), prevO(r_idx));
                end
            end  
    
            % compute mean values by each batch
            EntFields = fieldnames(tempE.All);  
            for f = 1:length(EntFields)
                EntBlock_all(ne).(EntFields{f})    = mean(tempE.All.(EntFields{f}), 'omitnan');
                EntBlock_early(ne).(EntFields{f})  = mean(tempE.Early.(EntFields{f}), 'omitnan');
                EntBlock_before(ne).(EntFields{f}) = mean(tempE.beforeRev.(EntFields{f}), 'omitnan');
                EntBlock_steady(ne).(EntFields{f}) = mean(tempE.Steady.(EntFields{f}), 'omitnan');
                if inclue_rev_flag
                    EntBlock_rev(ne).(EntFields{f}) = mean(tempE.afterRev.(EntFields{f}), 'omitnan');
                end
            end        
    
            % calculate metric for each trial position
            tempTrial = struct;
            for t = 1:blockL-1
                stay  = CompStat.stay(:,t);            
                prevR = CompStat.prevR(:,t);
                prevO = CompStat.prevO(:,t);                
                currC = CompStat.currC(:,t); % current choice, shifted by 1 to match index for strategy
                hr_opt = CompStat.hr_opt(:,t); % also shifted                
                [tempTrial] = compute_behavioral_and_entropy_met(tempTrial, decomp_on, currC, prevR, hr_opt, stay, prevR, prevO);
            end 
            % reassign data to struct array 
            for f = 1:length(EntFields)                    
                EntTrial(ne).(EntFields{f}) = tempTrial.(EntFields{f})';
            end
            EntTrial(ne).pbetter = [mean(CompStat.firstC), EntTrial(ne).pbetter]; % concat. performance from first trial
            EntTrial(ne).pwin(1,blockL) = mean(CompStat.lastR); % concat. win rate from last trial
        end   
        ModOutput{m}.Trial = EntTrial;
        ModOutput{m}.Block.All = EntBlock_all;
        ModOutput{m}.Block.Early = EntBlock_early;
        ModOutput{m}.Block.Before = EntBlock_before;
        ModOutput{m}.Block.Steady = EntBlock_steady;
        if inclue_rev_flag
            ModOutput{m}.Block.afterRev = EntBlock_rev;
        end
        toc
    end
    disp("Sim complete ("+toc(allSt)/60+" min)");
end