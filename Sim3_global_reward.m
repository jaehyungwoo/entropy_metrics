% Simulation 3: Revealing adjustments in behavioral strategies driven by global reward rate
clc; close all

% set simulation params
alphaQ = 0.3;   
alphaG = 0.3;  
beta1  = 10;  % inv. temp.
beta_G = 2;   % influence of G

% initialize models
models = init_sim_models([alphaQ,beta1,alphaG,beta_G]);

%% Specify reward environment: 80/20, L = 80
rew_probs = [0.8, 0.2];
blockL = 80;
rev_pos = 40;
RewProbs = Construct_reward_schedule(rew_probs, blockL, rev_pos);
EnvLbl = max(RewProbs(1,:))*100+"/"+min(RewProbs(1,:))*100+"%";

FontS = 15;

%% Simulate agent behavior
numEnv = 100; % number of env. (sim. batch)
numSim = 100;  % number of repeated sim per environment

ModOutput = cell(1,length(models));
for m = 1:length(models)
    disp(m+"."+models{m}.name);
    player = struct;
    player.label = strcat('sim', models{m}.name);
    player.params = models{m}.simpar;  

    % initialize variable
    clear EntTrial EntBlock_all
    EntTrial(numEnv) = struct; % across trials w.r.t. rev    
    EntBlock_all(numEnv) = struct; % within blocks (all trials)

    ModOutput{m}.Trial = struct;
    ModOutput{m}.Block = struct;
    
    % loop through each batch
    parfor nE = 1:numEnv
        CompStat = struct; % trial data to be compiled (for each batch)
        CompStat.stay  = []; % stay(1) or switch(0)
        CompStat.prevR = []; % rewarded(1) or not(0)
        CompStat.prevO = []; % chose better(0) or not(0)
        CompStat.prevG = [];
        CompStat.prevRG = []; % Reward & GRS
        CompStat.V_t = [];
        CompStat.Q1 = [];
        CompStat.Q2 = [];
        CompStat.currC = [];
        CompStat.hr_opt = [];
        
        % repeat random simulation of each environment   
        tempE = struct;
        simEnv = SetUp_RewardEnvironment(RewProbs, nE); 
        for ns = 1:numSim
            % simulate task with given seed:            
            % this randomizes choice and reward generation
            simStats = Simulate_ModelChoice_randR(player, simEnv, 'AB', ns+(nE-1)*numSim); 

             % reward rate V
            if isfield(simStats,'grr')
                CompStat.V_t = [CompStat.V_t; simStats.grr'];
                GRS = simStats.grr >= median(simStats.grr);
            else
                V_t = 0.5;
                for t = 2:length(simStats.r)
                    V_t(t) = V_t(t-1) + alphaG*(simStats.r(t-1)-V_t(t-1));
                end
                CompStat.V_t = [CompStat.V_t; V_t];
                GRS = V_t' >= median(V_t);
            end
            CompStat.Q1 = [CompStat.Q1; simStats.q1'];
            CompStat.Q2 = [CompStat.Q2; simStats.q2'];
            stay  = simStats.c(1:end-1)==simStats.c(2:end);
            prevR = simStats.r(1:end-1);
            prevO = simStats.c(1:end-1)==simEnv.hr_stim(1:end-1); 
            prevG = GRS(1:end-1);
            prevRO = binary_to_decimal([prevR, prevO]);
            prevRG = binary_to_decimal([prevR, prevG]);

            % concatenate same trial data (for across-trial data)        
            CompStat.stay =  [CompStat.stay; stay']; % [blocks x trials]
            CompStat.prevR = [CompStat.prevR; prevR'];
            CompStat.prevO = [CompStat.prevO; prevO'];
            CompStat.prevG = [CompStat.prevG; prevG'];
            CompStat.prevRO = [CompStat.prevRO; prevRO'];
            CompStat.prevRG = [CompStat.prevRG; prevRG'];
            CompStat.currC = [CompStat.currC; simStats.c(2:end)'];
            CompStat.hr_opt = [CompStat.hr_opt; simEnv.hr_stim(2:end)'];

             % compute within-block metrics
            [tempE] = GRR_behavioral_and_entropy_met(tempE, simStats.c, simStats.r, simEnv.hr_stim, stay, prevR, prevO, prevG);
        end
        % compute mean values by each batch
        EntFields = fieldnames(tempE.All);
        for f = 1:length(EntFields)
            EntBlock_all(nE).(EntFields{f}) = mean(tempE.All.(EntFields{f}), 'omitnan');
        end     

        % calculate metric for each trial position
        tempTrial = struct;
        for t = 1:blockL-1
            stay  = CompStat.stay(:,t);
            prevR = CompStat.prevR(:,t);
            prevG = CompStat.prevG(:,t);
            prevO = CompStat.prevO(:,t);
            prevRO = CompStat.prevRO(:,t);
            prevRG = CompStat.prevRG(:,t);
            currC = CompStat.currC(:,t); % current choice, shifted by 1 to match index for strategy
            hr_opt = CompStat.hr_opt(:,t); % also shifted

            [tempTrial] = GRR_behavioral_and_entropy_met(tempTrial, currC, prevR, hr_opt, stay, prevR, prevO, prevG);
        end
        for f = 1:length(EntFields)
            EntTrial(nE).(EntFields{f}) = tempTrial.(EntFields{f})';
        end
        EntTrial(nE).GRR = mean(CompStat.V_t,1,'omitnan'); 
        EntTrial(nE).Q1  = mean(CompStat.Q1,1,'omitnan'); 
        EntTrial(nE).Q2  = mean(CompStat.Q2,1,'omitnan'); 
    end
    ModOutput{m}.Trial = EntTrial;
    ModOutput{m}.Block = EntBlock_all;
end
disp("Sim complete");


%% Plot example time course of metrics (across-block metrics)

%%% choose metric to plot on Y-axis:

% met_set = "H_str"; met_lbl = "H(Str)";         % etc.
% met_set = "ERDS"; met_lbl = met_set;           % etc.
% met_set = "n_MIRS"; met_lbl = "n-MIRS";        % etc.
% met_set = "n_MIRGS"; met_lbl = "n-MIRGS";      % fig.5c

met_set = "n_MIRGS_high"; met_lbl = "n-MIRGS_{high}";      % fig.5d
% met_set = "n_MIRGS_winhigh"; met_lbl = "n-MIRGS_{high+}";      % etc.
% met_set = "n_MIRGS_losehigh"; met_lbl = "n-MIRGS_{high-}";      % etc.
met_set = "n_MIRGS_low"; met_lbl = "n-MIRGS_{low}";      % fig.5e
% met_set = "n_MIRGS_winlow"; met_lbl = "n-MIRGS_{low+}";      % etc.
% met_set = "n_MIRGS_loselow"; met_lbl = "n-MIRGS_{low-}";      % etc.

figure(30); clf
set(gcf,'Color','w','Units','normalized','Position',[0, 0, 0.27, 0.2]); hold on
for mod = 1:length(models)
    if isfield(ModOutput{mod}.Trial, met_set)
        this_dat = reshape([ModOutput{mod}.Trial.(met_set)], [], numSim)'; 
    else
        switch met_set
            case "n_MIRGS_high"
                win_high = reshape([ModOutput{mod}.Trial.n_MIRGS_winhigh], [], numSim)'; 
                lose_high = reshape([ModOutput{mod}.Trial.n_MIRGS_losehigh], [], numSim)'; 
                this_dat = win_high + lose_high;
            case "n_MIRGS_low"
                win_low = reshape([ModOutput{mod}.Trial.n_MIRGS_winlow], [], numSim)'; 
                lose_low = reshape([ModOutput{mod}.Trial.n_MIRGS_loselow], [], numSim)'; 
                this_dat = win_low + lose_low;
        end
    end
    shadedErrorBar(2:blockL, mean(this_dat,1,'omitnan'), sem(this_dat,1),'lineprops',{'Color',mod_cols{mod}}); 
end
xline(rev_pos+.5,'--k','LineWidth',1, 'HandleVisibility','off');    
xlabel("Trials"); xlim([1 blockL]);
ylabel(met_lbl); 
set(gca,'FontSize',FontS,'TickDir','out','box','off','LineWidth',0.75);
legend("\alpha = "+alpha_set,'linewidth',.5,'box','off');

%% Within-block metrics (insets)

figure(31); clf; 
set(gcf,'Color','w','Units','normalized','Position',[0, 0, 0.08, 0.08]);
allDat = cell(1,length(models));
for mod = 1:length(models)    
    if isfield(ModOutput{mod}.Trial, met_set)
        this_dat = [ModOutput{mod}.Block.(met_set)];
    else
        switch met_set
            case "n_MIRGS_high"
                this_dat = [ModOutput{mod}.Block.n_MIRGS_winhigh] + [ModOutput{mod}.Block.n_MIRGS_losehigh];
            case "n_MIRGS_low"
                this_dat = [ModOutput{mod}.Block.n_MIRGS_winlow] + [ModOutput{mod}.Block.n_MIRGS_loselow];
        end
    end
    B = bar(mod, mean(this_dat,'omitnan'), 'BarWidth',0.9, 'EdgeColor','none'); hold on;
    B.FaceColor = mod_cols(mod,:);
    errorbar(mod, mean(this_dat,'omitnan'), sem(this_dat,2), 'Color',[.5 .5 .5]);
    allDat{mod} = this_dat;
end
xticks(1:length(models)); xticklabels([]);
set(gca,'FontSize',FontS,'TickDir','out','box','off','LineWidth',.5);

%% Full parameter sweep (data for Fig.5f)
alpha_set = .1:.2:.9;
betaG_set = 0:.5:3;

numEnv = 100;
numSim = 1;

models = init_sim_models([alphaQ,beta1,alphaG,beta_G]);
SimMod = models{2};

GridDat = cell(length(betaG_set),length(alpha_set));
for q1 = 1:length(betaG_set)
    betaG = betaG_set(q1);      disp(" >> "+q1+"/"+length(betaG_set));
    for q2 = 1:length(alpha_set)
        alphaQ = alpha_set(q2);  disp(q2+"/"+length(alpha_set));
        player = struct;
        player.label = strcat('sim', SimMod.name);
        player.params = [alphaQ, beta1, alphaQ, betaG];  

        % initialize variable
        clear EntBlock_all   
        EntBlock_all(numEnv) = struct; % within blocks (all trials)
        GridDat{q1,q2} = struct;

        % loop through each batch
        tic
        parfor nE = 1:numEnv
            % repeat random simulation of each environment   
            tempE = struct;
            simEnv = SetUp_RewardEnvironment(RewProbs, nE); 
            for ns = 1:numSim           
                simStats = Simulate_ModelChoice_randR(player, simEnv, 'AB', ns+(nE-1)*numSim); 
                GRS = simStats.grr >= median(simStats.grr);
                stay  = simStats.c(1:end-1)==simStats.c(2:end);
                prevR = simStats.r(1:end-1);
                prevG = GRS(1:end-1);
                prevRG = binary_to_decimal([prevR, prevG]);
                tempE = append_to_fields(tempE, { Mutual_Information_weighted(stay, prevRG, "MIRGS",containers.Map({0,1,2,3},{'loseLow','loseHigh','winLow','winHigh'})),...
                    Mutual_Information_weighted(stay, prevG, "MIGS", containers.Map({0,1},{'low','high'})) });
            end  
    
            % compute mean values by each batch      
            EntFields = fieldnames(tempE);
            for f = 1:length(EntFields)
                EntBlock_all(nE).(EntFields{f}) = mean(tempE.(EntFields{f}), 'omitnan');
            end
        end   
        GridDat{q1,q2} = EntBlock_all;
        toc
    end
    disp("Sim complete");
end

%% Fig. 5f: plot line graphs of results

% met_set = "n_MIRGS";                % etc.
% met_set = "N_pMIRGS_winhigh";       % left, solid
% met_set = "N_pMIRGS_losehigh";    % left, dashed
% met_set = "N_pMIRGS_winlow";      % right, solid
met_set = "n_MIRGS_loselow";      % right, dashed

figure(36); clf
set(gcf,'Color','w','Units','normalized','Position',[0, 0, 0.18, 0.2]);
cmap = cool(length(betaG_set));
allDat = cell(1,length(met_set));
for m = 1:length(met_set)
    allDat{m} = [];
    for q1 = 1:length(betaG_set)
        meanDat = nan(1, length(alpha_set));
        semDat  = nan(1, length(alpha_set));    
        for q2 = 1:length(alpha_set)
            ydat = assign_output(GridDat{q1,q2}, met_set(m), "phase", "all");        
            meanDat(q2) = mean(ydat,'omitnan');
            semDat(q2)  = sem(ydat);
            allDat{m} = [allDat{m}; ydat(:), repmat(betaG_set(q1), length(ydat), 1), repmat(alpha_set(q2), length(ydat), 1)];
        end
        errorbar(alpha_set, meanDat, semDat, lin_type(m),'Color',cmap(q1,:)); hold on;
    end
    % regression on beta_G & alpha
    sA = normalize(allDat{m});
    mdl = fitlm(sA(:,2:3), sA(:,1), "interactions"); disp(mdl);
end
xlabel("\alpha"); xticks(alpha_set);
ylabel(met_lbl);
yline(0,":k",'HandleVisibility','off');
legend("\beta_G = "+betaG_set,'location','eastoutside','NumColumns',1);
set(gca,'FontSize',FontSize,'TickDir','out','box','off','LineWidth',.75);


%% Function: initialize models to simulate
function [models] = init_sim_models(params)
    models = {};    % initialize as cell array

    alpha_Q = params(1);
    beta_1  = params(2);
    alpha_G = params(3);
    beta_G  = params(4);

    % (2-alpha model)
    m = length(models) + 1;
    models{m}.name = 'RL_2alpha';      
    models{m}.fun = 'simRL_2alpha'; % specify simulation function   
    models{m}.simpar  =[alpha_Q, beta_1, alpha_Q];       
    models{m}.lb      =[ 0   1  0];   
    models{m}.ub      =[ 1 100  1];   
    models{m}.label = "RL";    
    models{m}.plabels = ["\alpha+", "\beta", "\alpha-"];

    %2. GRR-based
    m = length(models) + 1;
    models{m}.name = 'RL_deltaGRRstay';      
    models{m}.fun = 'simRL_deltaGRRstay'; % specify simulation function       
    models{m}.lb    =[ 0   1  0  3];   
    models{m}.ub    =[ 1 100  1  3]; 
    models{m}.simpar=[alpha_Q  beta_1 alpha_G beta_G];
    models{m}.label = "GRR-based (\delta)";    
    models{m}.plabels = ["\alpha_Q", "\beta_1", "\alpha_V", "\beta_G"];

    % load index
    if ~exist('idx','var')
        idx = 1:length(models);
    end
    models = models(idx);
    for m = 1:length(models)
        disp(m+". "+models{m}.label);
    end
end
%% subfunc
function [MetOut] = GRR_behavioral_and_entropy_met(MetIn, choice, reward, hr_opt, stay, prevR, prevO, prevG)
    R_decomp = containers.Map({0,1},{'lose','win'});
    O_decomp = containers.Map({0,1},{'worse','better'});
    RO_decomp = containers.Map({0,1,2,3},{'loseworse','losebetter','winworse','winbetter'});
    RG_decomp = containers.Map({0,1,2,3},{'loselow','losehigh','winlow','winhigh'});
    
    % obtain vectors for computing entropy measures            
    prevRO  = binary_to_decimal([prevR, prevO]);
    prevRG  = binary_to_decimal([prevR, prevG]);
    currBW = choice==hr_opt;
    
    % intialize and compute performance-based metrics
    MetOut = struct;
    MetOut.pwin = mean(reward);
    MetOut.pbetter = mean(currBW);

    % stay decompositions
    MetOut.pStay       = mean(stay);
    MetOut.pStayWin    = mean(stay & prevR) / mean(prevR);
    MetOut.pSwitchLose = mean(~stay & ~prevR) / mean(~prevR);

    % entropy of strategy
    MetOut.H_str = Shannon_Entropy(stay);    
    MetOut = append_to_fields(MetOut, {Conditional_Entropy_decomp(stay, prevR, "ERDS", R_decomp ), ...                                    
                                    Mutual_Information_decomp(stay, prevR, "MIRS", R_decomp ), ...                                    
                                    Conditional_Entropy_decomp(stay, prevO, "EODS", O_decomp ), ...
                                    Mutual_Information_decomp(stay, prevO, "MIOS", O_decomp), ...
                                    Conditional_Entropy_decomp(stay, prevRO, "ERODS", RO_decomp), ...
                                    Mutual_Information_decomp(stay, prevRO, "MIROS", RO_decomp), ...
                                    Conditional_Entropy_decomp(stay, prevRG, "ERGDS", RG_decomp), ...
                                    Mutual_Information_decomp(stay, prevRG, "MIRGS", RG_decomp) });
    MetOut = append_to_fields(MetIn, {MetOut});
end

%
function struct1 = append_to_fields(struct1, struct2s)
    for i = 1:length(struct2s)
        struct2 = struct2s{i};
        for fields = fieldnames(struct2)'
            if isfield(struct1, fields{1})
                struct1.(fields{1}) = [struct1.(fields{1}); struct2.(fields{1})];
            else
                struct1.(fields{1}) = struct2.(fields{1});
            end
        end
    end
end