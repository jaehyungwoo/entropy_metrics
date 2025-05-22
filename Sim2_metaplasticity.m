% Simulation 2: Revealing reward-dependent metaplasticity
close all; clc
clearvars; % addpath(genpath(pwd));

FontS = 15;
%% Specify params
alph1 = 0.3;    % learning rate    
beta1 = 10;     % inv. temp.
mp1   = 0.4;    % meta-transition rate

% initialize models
models = init_sim_models([alph1, beta1, mp1, alph1]);

R_decomp = containers.Map({0,1},{'lose','win'});
O_decomp = containers.Map({0,1},{'worse','better'});
RO_decomp = containers.Map({0,1,2,3},{'loseworse','losebetter','winworse','winbetter'});

%% Specify reward environment: 80/20, L = 80
rew_probs = [0.8, 0.2];
blockL = 80;
rev_pos = 40;
RewProbs = Construct_reward_schedule(rew_probs, blockL, rev_pos);
EnvLbl = max(RewProbs(1,:))*100+"/"+min(RewProbs(1,:))*100+"%";

%% 1. Plot simulated signals: comaprison with plastic RL
close all
numSim = 1000;  % number of simulation
ModLines = [":","-"];
figure(40); clf; 
set(gcf,'Color','w','Units','normalized','Position',[0, 0, 0.27, 0.2]);
ModLabels = strings;
ModOutput = cell(1,length(models));

for m = 1:length(models)
    player = struct;
    player.label = strcat('sim', models{m}.name);
    player.params = models{m}.simpar;  
    CompStat = struct;
        CompStat.q1 = []; CompStat.q2 = [];
        CompStat.eff_plus = []; CompStat.eff_minus = [];
        CompStat.str_syn1 = []; CompStat.str_syn2 = []; % strong synapses
        CompStat.wk_syn1  = []; CompStat.wk_syn2 = []; % weak synapses

    % repeat random simulation of environment 
    for nn = 1:numSim
        simEnv   = SetUp_RewardEnvironment(RewProbs, nn);
        simStats = Simulate_ModelChoice_randR(player, simEnv, 'AB', nn);               

        Q = [simStats.q1, simStats.q2];
        Q_c = Q(sub2ind(size(Q), (1:blockL)', (simStats.c/2)+1.5)); % Q_chosen
        DeltaQ = [diff(Q,[],1); nan(1,2)];
        DeltaQc = DeltaQ(sub2ind(size(Q), (1:blockL)', (simStats.c/2)+1.5)); % deltaQ_chosen
        
        eff_alpha = DeltaQc ./ (simStats.r - Q_c); % effective learning rate    
        eff_plus  = eff_alpha; eff_plus(simStats.r==0) = NaN;
        eff_minus = eff_alpha; eff_minus(simStats.r==1) = NaN;

        % concatenate same trial data        
        CompStat.q1 = [CompStat.q1; simStats.q1']; % [blocks x trials]
        CompStat.q2 = [CompStat.q2; simStats.q2'];    
        CompStat.eff_plus = [CompStat.eff_plus; eff_plus'];
        CompStat.eff_minus = [CompStat.eff_minus; eff_minus'];
    end
    ModOutput{m} = CompStat;
    V1 = mean(ModOutput{m}.q1,1,'omitnan');
    shadedErrorBar(1:blockL, V1, sem(ModOutput{m}.q1,1),'lineprops',{'Color','#006837','linestyle',ModLines{m}}); hold on;
    V2 = mean(ModOutput{m}.q2,1,'omitnan');
    shadedErrorBar(1:blockL, V2, sem(ModOutput{m}.q2,1),'lineprops',{'Color','#F7931E','linestyle',ModLines{m}, 'handlevisibility','off'});
    ModLabels(m) = models{m}.label;
end
legend(ModLabels,'box','off');
xline(rev_pos+.5,':k','LineWidth',1,'HandleVisibility','off');
ylim([0 1]);
ylabel("\langleV_i\rangle");
xlabel("Trials");
set(gca,'FontSize',FontS,'TickDir','out','box','off','LineWidth',1);

%% 2. Plot estimated effective learning rates
figure(41); clf;
set(gcf,'Color','w','Units','normalized','Position',[0, 0, 0.27, 0.25]);
ModLabels = strings; cnt = 0;
for m = 1:length(ModOutput)
    V1 = mean(ModOutput{m}.eff_plus,1,'omitnan');
    shadedErrorBar(1:length(V1), V1, sem(ModOutput{m}.eff_plus,1),'lineprops',{'Color','r','linestyle',ModLines{m}}); hold on;
    V2 = mean(ModOutput{m}.eff_minus,1,'omitnan');
    shadedErrorBar(1:length(V2), V2, sem(ModOutput{m}.eff_minus,1),'lineprops',{'Color','b','linestyle',ModLines{m}, 'handlevisibility','off'});
end
xline(rev_pos+.5,'--k','LineWidth',1,'HandleVisibility','off');
xticks(10:10:blockL);
ylabel("effective \alpha"); 
ylim([0 0.5]); yticks(0:.1:0.5);
xlabel("Trials");
set(gca,'FontSize',FontS,'TickDir','out','box','off','LineWidth',.75);

%%  3. Simulated entropy metrics, both time course and within-blocks

numEnv = 100; % number of env. (sim. batch)
numSim = 100; % number of repeated sim per environment
steadyL  = 10;
afterRev = 10;

[ModAllOutput] = simulate_entropy_metrics_models(models, RewProbs, numEnv, numSim, steadyL, rev_pos, afterRev);

%% 4. Fit 2-alpha RL to choice behavior of RDMP
decomp_on = 1;
fit_beta_flag = 0;

% optimize options
numFit = 10;
op = optimset('fminsearch');
op.MaxIter = 1e7; op.MaxFunEvals = 1e7;

allSt = tic;
% initialize variable
clear EntTrial EntBlock_all EntBlock_early EntBlock_steady EntBlock_rev   
EntTrial(numEnv) = struct; % across trials w.r.t. rev    
EntBlock_all(numEnv) = struct; % within blocks (all trials)
parfor ne = 1:numEnv
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

    simEnv = SetUp_RewardEnvironment(RewProbs, ne); 
    tempE = struct;
    tempE.All = struct;
    tempE.Steady = struct;
    tempE.beforeRev = struct;
    CompStat = struct;
    CompStat.stay  = [];
    CompStat.prevR = []; % rewarded(1) or not(0)
    CompStat.prevO = [];
    CompStat.currC = [];
    CompStat.hr_opt = [];
    CompStat.eff_plus = []; CompStat.eff_minus = []; % also store effective learnign rates
    for ns = 1:numSim
        tempStats = Simulate_ModelChoice_randR(SimMod, simEnv, 'AB', ns+(ne-1)*numSim);

        % fit RL model
        qpar = cell(numFit,1); NegLL = nan(numFit,1);
        for ii = 1:numFit
            [qpar{ii}, NegLL(ii)] = fmincon(fitfunc_handle, rand(1,length(lb)), [], [], [], [], lb, ub, [], op, {tempStats.c, tempStats.r, beta1});
        end
        fpar = qpar(min(NegLL)==NegLL);   
        
        % use fitted RL to simulate the data
        FitMod.params = [fpar{1}(1), beta1, fpar{1}(2)];
        simStats = Simulate_ModelChoice_randR(FitMod, simEnv, 'AB', ns+(ne-1)*numSim);
        stay  = simStats.c(1:end-1)==simStats.c(2:end);
        prevR = simStats.r(1:end-1); 
        prevO = simStats.c(1:end-1)==simEnv.hr_stim(1:end-1);
        CompStat.stay =  [CompStat.stay; stay']; % [blocks x trials]
        CompStat.prevR = [CompStat.prevR; prevR'];
        CompStat.prevO = [CompStat.prevO; prevO'];        
        CompStat.currC = [CompStat.currC; simStats.c(2:end)'];
        CompStat.hr_opt = [CompStat.hr_opt; simEnv.hr_stim(2:end)'];
        % compute whole-block metrics
        [tempE.All] = compute_behavioral_and_entropy_met(tempE.All, decomp_on, simStats.c, simStats.r, simEnv.hr_stim, stay, prevR, prevO);

        % N trials before reversal
        s_idx = rev_pos-steadyL+1:rev_pos;
        [tempE.beforeRev] = compute_behavioral_and_entropy_met(tempE.beforeRev, decomp_on, simStats.c(s_idx+1), simStats.r(s_idx+1), simEnv.hr_stim(s_idx+1), stay(s_idx), prevR(s_idx), prevO(s_idx));

        % steady state: last N trials within each phase
        s_idx = [rev_pos-steadyL+1:rev_pos, blockL-steadyL+1:blockL]-1;
        s_idx = s_idx(~isnan(s_idx));
        [tempE.Steady] = compute_behavioral_and_entropy_met(tempE.Steady, decomp_on, simStats.c(s_idx+1), simStats.r(s_idx+1), simEnv.hr_stim(s_idx+1), stay(s_idx), prevR(s_idx), prevO(s_idx));

        CompStat.eff_plus = [CompStat.eff_plus; ones(1,blockL-1)*fpar{1}(1)];
        CompStat.eff_minus = [CompStat.eff_minus; ones(1,blockL-1)*fpar{1}(2)];
    end

    % compute mean values by each batch
    EntFields = fieldnames(tempE.All);  
    for f = 1:length(EntFields)
        EntBlock_all(ne).(EntFields{f}) = mean(tempE.All.(EntFields{f}), 'omitnan');
        EntBlock_steady(ne).(EntFields{f}) = mean(tempE.Steady.(EntFields{f}), 'omitnan');
        EntBlock_rev(ne).(EntFields{f}) = mean(tempE.beforeRev.(EntFields{f}), 'omitnan');        
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
    for f = 1:length(EntFields)                    
        EntTrial(ne).(EntFields{f}) = tempTrial.(EntFields{f})';
    end
    % store effective learn. rates
    EntTrial(ne).eff_plus = mean(CompStat.eff_plus, 1,'omitnan');
    EntTrial(ne).eff_minus = mean(CompStat.eff_minus, 1,'omitnan');
end
RefitOutput.Trial = EntTrial;
RefitOutput.Block.All = EntBlock_all;
RefitOutput.Block.Steady = EntBlock_steady;
RefitOutput.Block.Before = EntBlock_rev;

disp("Sim & fit complete ("+toc(allSt)/60+" min)");
ModAllOutput{3} = RefitOutput;
models{3} = models{1};
models{3}.label = "RL refit";

%% 5. Plot estimated effective learning rates of re-fitted RL
ModOutput{3} = struct;
ModOutput{3}.eff_plus = reshape([RefitOutput.Trial.eff_plus], [], numEnv)';
ModOutput{3}.eff_minus = reshape([RefitOutput.Trial.eff_minus], [], numEnv)';

% close all
ModLines = [":",'-',"--"];

figure(45); clf;
set(gcf,'Color','w','Units','normalized','Position',[0, 0, 0.27, 0.25]); hold on;
ModLabels = strings; cnt = 0;
for m = 1:length(ModOutput)
    cnt = cnt + 1;
    ModLabels(cnt) = models{m}.label;

    V1 = mean(ModOutput{m}.eff_plus,1,'omitnan');    
    V2 = mean(ModOutput{m}.eff_minus,1,'omitnan');
    if contains(models{m}.label,"RDMP")
        shadedErrorBar(1:length(V1), V1, sem(ModOutput{m}.eff_plus,1),'lineprops',{'Color','r','linestyle',ModLines{m}});
        shadedErrorBar(1:length(V2), V2, sem(ModOutput{m}.eff_minus,1),'lineprops',{'Color','b','linestyle',ModLines{m}, 'handlevisibility','off'});
    else%if m==1
        plot(1:length(V1), V1, 'r','linestyle',ModLines{m});
        plot(1:length(V2), V2, 'b','linestyle',ModLines{m}); 
    end
end
legend(ModLabels,'box','off');
xline(rev_pos+.5,'--k','LineWidth',1,'HandleVisibility','off');
xticks(10:10:blockL);
ylabel("effective learning rates"); ylim([0 alph1*2]); yticks(0:.1:alph1*2);
xlabel("Trials");
set(gca,'FontSize',FontS,'TickDir','out','box','off','LineWidth',.75);

%% 6. Plot simulated metrics
mod_cols = {'k','m',[.6 .6 .6]};

figure(46); clf
set(gcf,'Color','w','Units','normalized','Position',[0, 0, 0.27, 0.25]);
for m = 1:length(ModAllOutput)
    this_dat = reshape([ModAllOutput{m}.Trial.(met_set)], [], numEnv)';
    shadedErrorBar(1:size(this_dat,2), mean(this_dat,1,'omitnan'), sem(this_dat,1),'lineprops',{'Color',mod_cols{m}, 'linewidth',1.5});
    hold on;            
    xlabel("Trials"); xlim([1 blockL]);
    ModLabels(m) = models{m}.label;
    % ylim([0 1]); 
    xlim([1 blockL]);
    ylabel(strjoin(met_lbl,", "));
    if prod(unique(sign(mean(this_dat,1,'omitnan'))),'omitnan')==-1
        yline(0,":k",'HandleVisibility','off')
    end 
end
xline(rev_pos+.5,'--k','LineWidth',1, 'HandleVisibility','off');
legend(ModLabels, 'box','off','location','northwest');
set(gca,'FontSize',FontS,'TickDir','out','box','off','LineWidth',.75);

%% 7. Decoding of positivity bias

DecodeVar_set = {'all'};    % use all metrics to maximize accuracy
DecodeVar_lbl = "All metrics";

% Initialize decoding class
numEnv = 100;
numSim = 100;    % blocks within sess
   
% structure containing decoding options
DecodeOpt = struct; 
DecodeOpt.train_type = "Trial"; % metrics comptued from non-overlapping block      
DecodeOpt.metric_type = "raw";
DecodeOpt.sample_type = "grid";         % 'grid' or 'random' 
DecodeOpt.comp_group = "pessimistic";   % vs. optimistic
DecodeOpt.VarSet        = DecodeVar_set; % set of used features (variables)
DecodeOpt.VarLbl        = DecodeVar_lbl;
DecodeOpt.RewProbs      = RewProbs;
DecodeOpt.numEnv = numEnv;
DecodeOpt.numSim = numSim;
DecodeOpt.sessfit_flag = 1;  % flag for fitting RL to RDMP behavior from each session
DecodeOpt.conv_nan2zero = 1; % flag for converting NaN metric values to zeros

% initialize
decoder = RDMP_Decoder_class(DecodeOpt);

%% Obtain (simulate) test data from RL and RDMP
alph0   = 0.3;   % base learning rate for RL & RDMP
mp1     = 0.4;   % base meta-transition prob. for RDMP
beta1   = 10;    % fixed inv. temp.

[ModOutput] = decoder.obtain_test_data([alph0, beta1, mp1]);
disp("Test data obtained");

params = struct;
params.alpha_range = [0.2 0.4];
params.stepSize = 0.01;
params.minDiff = 0.03;

%% Run decoding experiments N times
sampleN = 1; % # of experiments

PostProb = cell(sampleN,1);
AccuMAT  = cell(sampleN,1);
for Sd = 1:sampleN   
    sStart = tic;

    % training decoder w/ sampled alpha's
    [decoder, sampleDat] = decoder.train_decoder(params, beta1, Sd);
    
    % testing on model simulated metrics
    [PostProb{Sd}, AccuMAT{Sd}] = decoder.test_decoder(ModOutput);

    disp("------------[Seed "+Sd+"] "+toc(sStart)/60+" min------------");
end
disp("Complete");

%% plot results (Fig.4f)
mod_cols = {'k','m',[.6 .6 .6]};

figure(46); clf 
set(gcf,'Color','w','Units','normalized','Position',[0, 0, 0.4, 0.3]); hold on;
hold on;
x_trials = 1:blockL-1;
numTrials = size(PostProb{1}{1,1},2);
for m = 1:length(mod_cols)
    DatMeans = cell(length(PostProb),1);
    if 1
        DatMAT = PostProb;    ylbl = "P(optimistic vs. pessimistic)";
    else
        DatMAT = AccuMAT;        ylbl = "Decoding accuracy";
    end
    for n = 1:length(PostProb)
        DatMeans{n} = DatMAT{n}{1,m};
    end
    DatMeans = cell2mat(DatMeans);
    shadedErrorBar(x_trials, mean(DatMeans,1,'omitnan'), sem(DatMeans,1),'lineprops',{'Color',mod_cols{m},'linewidth',1.5})
end
ylabel(ylbl);
xline(rev_pos+.5,'--k','LineWidth',1);
yline(0.5,":k"); 
ylim([0 1]); yticks(0:.2:1);
legend(["RL1","RDMP","RL2 (fitted to RDMP)"]);
set(gca,'FontSize',14,'TickDir','out','box','off','LineWidth',.75);

%% Function: initialize models to simulate
function [models] = init_sim_models(params, idx)
    models = {};    % initialize as cell array
        
    alph1 = params(1);
    beta1 = params(2);
    mp1   = params(3);
    alph2 = params(4);
    % dec1  = params(5);

    % find base alpha_1 for RDMP models which gives the same intial effective learning rate
    N_meta  = 4;         % # of meta-states
    f = @(x) mean(x .^ ( ( (N_meta-2)*(1:N_meta) + 1 ) / (N_meta-1))) - alph1;    
    options = optimoptions('fsolve', 'Display', 'off'); % 'iter' to show steps; use 'off' to suppress
    RDMP_a1 = fsolve(f, alph1, options);
    disp("RL learning rate: \alpha = "+alph1);
    disp("Corresponding RDMP q_1 = "+RDMP_a1);

    % (2-alpha model)
    m = length(models) + 1;
    models{m}.name = 'RL_2alpha';      
    models{m}.fun = 'simRL_2alpha'; % specify simulation function   
    models{m}.simpar  =[alph1, beta1, alph2];       
    models{m}.lb      =[ 0   1  0];   
    models{m}.ub      =[ 1 100  1];   
    models{m}.label = "RL";    
    models{m}.plabels = ["\alpha+", "\beta", "\alpha-"];

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