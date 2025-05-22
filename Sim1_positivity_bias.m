% Simulation 1. Revealing positivity bias
clearvars; clc

alph0 = 0.5;    % fixed learn rate (alpha- below)
beta1 = 10;     % inv. temp

% RL with two learning rates
SimMod.name = 'RL_2alpha';      
SimMod.fun  = 'simRL_2alpha'; % specify simulation function   
SimMod.simpar  =[alph0, beta1, alph0];       
SimMod.lb      =[ 0   1  0];   
SimMod.ub      =[ 1 100  1];   
SimMod.label = "RL2";    
SimMod.plabels = ["\alpha+", "\beta", "\alpha-"];

% Specify example environment: 80/20, L = 80
rew_probs = [0.8, 0.2];
blockL = 80;
rev_pos = 40;
RewProbs = Construct_reward_schedule(rew_probs, blockL, rev_pos);
EnvLbl = max(RewProbs(1,:))*100+"/"+min(RewProbs(1,:))*100+"%";

FontS = 15;

%% Simulate and plot metrics (for example alpha+ values w/ fixed alpha-)
numEnv = 100; % number of env. (sim. batch)
numSim = 100; % number of repeated sim per environment
steadyL  = 9;
afterRev = 10;

% specify set of alpha+ for simulations
alpha_set = 0.1:.2:0.9; 

models = repmat(SimMod,1,length(alpha_set));
for m = 1:length(models)
    models(m).simpar = [alpha_set(m), beta1 , alph0]; % alpha- is fixed
end
[ModOutput] = simulate_entropy_metrics_models(models, RewProbs, numEnv, numSim, steadyL, rev_pos, afterRev);

mod_cols = {'#ff00ff','#ff0000','#000000','#0000ff','#00ffff'};

%% Plot example time course of metrics (across-block metrics)

%%% choose metric to plot on Y-axis:

% met_set = "H_str"; met_lbl = "H(Str)";         % etc.
% met_set = "ERDS"; met_lbl = met_set;           % Fig.3d
% met_set = "n_MIRS"; met_lbl = "n-MIRS";        % Fig.3e
% met_set = "n_MIRS_win"; met_lbl = "n-MIRS+";   % etc.
met_set = "n_MIRS_lose"; met_lbl = "n-MIRS-";    % Fig.3g

figure(30); clf
set(gcf,'Color','w','Units','normalized','Position',[0, 0, 0.27, 0.2]); hold on
for mod = 1:length(models)
    this_dat = reshape([ModOutput{mod}.Trial.(met_set)], [], numSim)'; 
    shadedErrorBar(2:blockL, mean(this_dat,1,'omitnan'), sem(this_dat,1),'lineprops',{'Color',mod_cols{mod}}); 
end
xline(rev_pos+.5,'--k','LineWidth',1, 'HandleVisibility','off');    
xlabel("Trials"); xlim([1 blockL]);
ylabel(met_lbl); 
set(gca,'FontSize',FontS,'TickDir','out','box','off','LineWidth',0.75);
legend("\alpha = "+alpha_set,'linewidth',.5,'box','off');

%% Within-block metrics (insets)
Block_phase = "All";

figure(31); clf; 
set(gcf,'Color','w','Units','normalized','Position',[0, 0, 0.08, 0.08]);
allDat = cell(1,length(models));
for mod = 1:length(models)    
    this_dat = [ModOutput{mod}.Block.(Block_phase).(met_set)];    
    B = bar(mod, mean(this_dat,'omitnan'), 'BarWidth',0.9, 'EdgeColor','none'); hold on;
    B.FaceColor = mod_cols(mod,:);
    errorbar(mod, mean(this_dat,'omitnan'), sem(this_dat,2), 'Color',[.5 .5 .5]);
    allDat{mod} = this_dat;
end
xticks(1:length(models)); xticklabels([]);
set(gca,'FontSize',FontS,'TickDir','out','box','off','LineWidth',.5);

%% Correlation between DeltaAlpha and the metrics

% simulate the grid of alpha's
alphaSet = 0.1:.05:0.9; 

% will simulate and save output file
SimOutput = Run_and_Save_grid(SimMod, rew_probs, blockL, steadyL, rev_pos, steadyL, alpha_Set, beta1, numEnv, numSim);

%% draw scatter plots
Block_phase = "All";

%%% choose metric to plot on Y-axis:

% met_set = "pbetter"; met_lbl = "P(Better)";    % Fig.3c
% met_set = "ERDS"; met_lbl = met_set;           % Fig.3f
% met_set = "n_MIRS"; met_lbl = "n-MIRS";        % etc.
% met_set = "n_MIRS_win"; met_lbl = "n-MIRS+";   % etc.
met_set = "n_MIRS_lose"; met_lbl = "n-MIRS-";    % Fig.3h

xx = flip(alphaSet') - alphaSet;

figure(32); clf; 
set(gcf,'Color','w','Units','normalized','Position',[0, 0, 0.13, 0.2]);

yy = [SimOutput.(Block_phase).(met_set)]; %assign_metric_output(SimOutput, Block_phase, met_set(m));    
yy = flip(yy,1); % flip y-axis to match above alpha grid data; smaller alpha+ comes at lower rows
Sz = (75*repmat(aMinus_set,length(aMinus_set),1)); % marker sizes based on alpha- values
scatter(xx', yy', Sz', flip(cool(length(aPlus_set))), 'filled','markerfacealpha',0.75); hold on;

% correlation & least-square-fit line
CorrType = "Pearson";
% CorrType = "Spearman";
[rho,pval] = corr(xx(:), yy(:), 'type',CorrType);
r_txt = ["\it{r} \rm= "+num2str(rho,3),"\it{p} \rm= "+num2str(pval,4)];
text(1, 1, r_txt,'units','normalized','fontsize',16,'HorizontalAlignment','right','VerticalAlignment','top');
p = polyfit(xx(:),yy(:),1);
x = linspace(-.9,.9);
plot(x, polyval(p,x),"Color",[.5 .5 .5]);

xlabel("\Delta\alpha");
ylabel(met_lbl(m));
% ylim([0.5 1]);    
xline(0,":k");
xticks(-1:.5:1);
if sign(min(yy(:)))*sign(max(yy(:)))==-1
    yline(0,":k");
end
set(gca,'FontSize',FontS,'TickDir','out','box','off','LineWidth',.75);
sgtitle(Block_phase+" trials; L = "+size(RewProbs,1)+", "+EnvLbl);

%% Decoding positivity bias

EnvSet = {[.8 .2], [.7 .3], [.6 .4], [.5 .1], [.4 .1]};

for Q1 = 1:length(EnvSet)
    rew_probs = EnvSet{Q1};
    run_decoding_script;  % this will run and save output for each environment
end

%% compile results
varNum = length(DecodeVar_set);

AccuDat = cell(length(EnvSet), length(agentN_Set));
AllDat = AccuDat;
learner_opt = 1; % 1) logistic; 2) SVM; 3) logistic+rbf; 4) SMV+rbf
for Q1 = 1:length(EnvSet)
    rew_probs = EnvSet{Q1};
    for Q2 = 1:length(agentN_Set)
        agentN  = agentN_Set(Q2);
        fname = "output/decoding/AccuMAT_"+agentN+"agents_r"+radi+"_sig"+sigma+"_samp"+sampleN+"x"+withinN+"_P"+rew_probs(1)+"_P"+rew_probs(2)+"_L"+blockL+"_"+trial_portion+windowL+".mat";
        if exist(fname,'file')
            load(fname,'PosNeutral','PosNegative','DecodeOpt');
        else
            disp("("+Q1+","+Q2+"): File doesn't exists: "+fname);  
            continue;
        end
        tempPess1 = nan(sampleN, 4, varNum);
        tempPess2 = nan(sampleN, varNum);
        for n = 1:sampleN        
            tempPess1(n,:,:) = cell2mat(PosNegative.Kfold{n}')';
            tempPess2(n,:) = mean(cell2mat(PosNegative.LOO{n}'), 1, 'omitnan')';

            % Compile all data for stats
            AllDat{Q1,Q2} = [AllDat{Q1,Q2}; cell2mat(PosNegative.LOO{n}')];
        end
        tempPess1 = squeeze(tempPess1(:,learner_opt));

        % K-fold accuracy
        AccuDat{1}{Q1,Q2} = tempPess1;  % optimistic vs. pessimistic

        % Leave-one-out accuracy
        AccuDat{2}{Q1,Q2} = tempPess2;  % optimistic vs. pessimistic
    end    
end
disp("Load completed");

%% Plot results here
accu_opt = 1; % 1) K-fold; 2) Leave-one-out

var_to_disp = 1:length(DecodeVar_set);
disp(DecodeOpt.VarSet(var_to_disp)');

figure(accu_opt); clf
set(gcf,'Color','w','Units','normalized','Position',[0, 0, 0.16, 0.21]);

AccuBarDat = nan(length(EnvSet), length(var_to_disp));
SEMdat = nan(length(var_to_disp), length(EnvSet));
EnvLbls = strings(length(EnvSet),1);
for Q1 = 1:length(EnvSet)   
    cnt = 0;   
    for var_opt = var_to_disp
        cnt = cnt + 1;
        AccuBarDat(Q1, cnt) = mean(AccuDat{accu_opt}{Q1,1}(:,var_opt),'omitnan');
        SEMdat(Q1, cnt) = sem(AccuDat{accu_opt}{Q1,1}(:,var_opt),1);
        EnvLbls(Q1) = strjoin(string(EnvSet{Q1}*100),"/");
    end
end
B = bar(AccuBarDat,'EdgeColor','none','BarWidth',0.95); hold on;
B(1).FaceColor = ones(1,3)*.75;
B(2).FaceColor = ones(1,3)*.25;
errorbar(B(1).XEndPoints, AccuBarDat(:,1), SEMdat(:,1), 'linewidth',.5, 'LineStyle','none', 'Color','k');
errorbar(B(2).XEndPoints, AccuBarDat(:,2), SEMdat(:,2), 'linewidth',.5, 'LineStyle','none', 'Color',[.5 .5 .5]);
xticklabels(EnvLbls);
ylim([.5 1]); yticks(.5:.1:1);
xlabel("Reward environment");
ylabel("Decoding accuracy");
set(gca,'FontSize',FontS,'TickDir','out','box','off','LineWidth',.75);

%% Function for simulating grid 
function [SimOutput] = Run_and_Save_grid(SimMod, rew_probs, blockL, steadyL, rev_pos, revL, alpha_Set, beta1, numEnv, numSim)
    aPlus_set  = alpha_Set;
    aMinus_set = alpha_Set;

    block_address = [0, rev_pos, blockL];
    block_address(isnan(block_address)) = [];
    b_idx = cell(1,length(block_address)-1);
    for b = 1:length(block_address)-1
        b_idx{b} = block_address(b)+1:block_address(b+1);
    end
    better1_id = cell2mat(b_idx(1:2:end));
    better2_id = cell2mat(b_idx(2:2:end));
    RewProbs = nan(blockL,2);
    RewProbs(better1_id,1) = rew_probs(1); RewProbs(better2_id,1) = rew_probs(2);
    RewProbs(better2_id,2) = rew_probs(1); RewProbs(better1_id,2) = rew_probs(2);
    
    trial_info = struct;
    trial_info.rev_pos    = rev_pos;
    trial_info.steady_win = steadyL;
    trial_info.rev_win    = revL;

    disp("Simulating "+SimMod.label+", "+length(alpha_Set)+"-by-"+length(alpha_Set)+" grid");
    disp("  # of batch = "+numEnv+", # of sim = "+numSim);
    disp("  Fixing inv. temp. = "+beta1);
    
    SimOutput = struct;
    SimOutput.All    = struct;
    SimOutput.Early = struct;
    SimOutput.Steady = struct;
    SimOutput.afterRev = struct;
    
    fname = SimMod.label+"_"+numEnv+"x"+numSim+"_P"+rew_probs(1)+"_P"+rew_probs(2)+"_L"+blockL+"_beta"+beta1+"_"+length(aPlus_set)+"-by-"+length(aMinus_set)+"_steady"+steadyL;
    disp(fname);
    if ~isnan(rev_pos)
        fname = fname + "_rev"+trial_info.rev_win;
    end
    if exist(fname+".mat",'file')
        disp("File exists, skipping simulation...")
        return;
    end

    tStart = tic;
    for a1 = 1:length(alpha_Set)
        disp(a1+". alpha = "+alpha_Set(a1)); 
        tic
        for a2 = 1:length(alpha_Set)
            % specify model and parameters
            player = struct;
            player.label = strcat('sim', SimMod.name);
            player.params = [aPlus_set(a1), beta1, aMinus_set(a2)]; 
            
            % run sim & store output
            simOut = simulate_withinBlocks_metrics(player, RewProbs, numEnv, numSim, trial_info);                               
            MetFields = fieldnames(simOut.All);
            for ff = 1:length(MetFields)
                SimOutput.All.(MetFields{ff})(a1,a2) = simOut.All.(MetFields{ff});
                SimOutput.Early.(MetFields{ff})(a1,a2) = simOut.Early.(MetFields{ff});
                SimOutput.Steady.(MetFields{ff})(a1,a2) = simOut.Steady.(MetFields{ff});
                if ~isnan(rev_pos)
                    SimOutput.afterRev.(MetFields{ff})(a1,a2) = simOut.afterRev.(MetFields{ff});
                end
            end
        end
        toc
    end

    % Save output
    save("output/"+fname+".mat", 'SimOutput', 'RewProbs', 'trial_info');
    disp("File saved: "+fname);
    tEnd = toc(tStart);
    disp("Elapsed time for this file: "+tEnd/60+" minutes.");
end

%% subfunc
function [SimOutput] = simulate_withinBlocks_metrics(player, RewProbs, numEnv, numSim, trial_info)
% player: specifies model and simulation parameters
% numEnv: # of simulation batch
% numSim: # of simulation per batch
% trial_info: specifies reversal positions and trial window for 'steday state' and 'after rev' period
    rev_pos    = trial_info.rev_pos;
    steadyL    = trial_info.steady_win;
    afterRevL  = trial_info.rev_win;
        
    decomp_flag = 1;

    if isnan(rev_pos)
        inclue_rev_flag = false;
    else
        inclue_rev_flag = true;
    end

    blockL = size(RewProbs, 1);

    % initialize struct array for parrallel comp.
    clear EntBlock_all EntBlock_steady EntBlock_rev           
    EntBlock_all(numEnv) = struct;   % within blocks (all trials)
    EntBlock_early(numEnv) = struct; % within blocks (first N trials only)
    EntBlock_steady(numEnv) = struct; % wihitn blocks (steady state only)
    EntBlock_rev(numEnv) = struct;    % wihitn blocks (after rev only)
    
    % tic 
    parfor ne = 1:numEnv
        % repeat random simulation of each environment   
        tempE = struct;
            tempE.All = struct;       % all trials
            tempE.Steady = struct;    % steady state
            tempE.afterRev = struct;       % 10 trials after rev
        
        % Set up unique environment for each batch
        simEnv = SetUp_RewardEnvironment(RewProbs, ne);

        for ns = 1:numSim 
            % simulate task with given seed:            
            % this randomizes choice and reward generation
            simStats = Simulate_ModelChoice_randR(player, simEnv, 'AB', ns+(ne-1)*numSim); 
            
            % obtain vectors for computing entropy measures
            stay  = simStats.c(1:end-1)==simStats.c(2:end);
            prevR = simStats.r(1:end-1);
            prevO = simStats.c(1:end-1)==simEnv.hr_stim(1:end-1);  

            % compute whole-block metrics
            [tempE.All] = compute_behavioral_and_entropy_met(tempE.All, decomp_flag, simStats.c, simStats.r, simEnv.hr_stim, stay, prevR, prevO);
            
            % early portion: first N trials 
            f_idx = 1:steadyL;
            [tempE.Early] = compute_behavioral_and_entropy_met(tempE.Early, decomp_flag, simStats.c(1:steadyL+1), simStats.r(1:steadyL+1), simEnv.hr_stim(1:steadyL+1), stay(f_idx), prevR(f_idx), prevO(f_idx));

            % steady state: N trials within each phase            
            s_idx = [rev_pos-(steadyL-1):rev_pos, blockL-(steadyL-1):blockL]-1;
            s_idx = s_idx(~isnan(s_idx));
            [tempE.Steady] = compute_behavioral_and_entropy_met(tempE.Steady, decomp_flag, simStats.c(s_idx+1), simStats.r(s_idx+1), simEnv.hr_stim(s_idx+1), stay(s_idx), prevR(s_idx), prevO(s_idx));

            % after reversal: N trials after reversals
            if inclue_rev_flag
                r_idx = (rev_pos+1:rev_pos+afterRevL) - 1;
                [tempE.afterRev] = compute_behavioral_and_entropy_met(tempE.afterRev, decomp_flag, simStats.c(r_idx+1), simStats.r(r_idx+1), simEnv.hr_stim(r_idx+1), stay(r_idx), prevR(r_idx), prevO(r_idx));
            end

        end

        % compute mean values by each batch
        EntFields = fieldnames(tempE.All);
        for f = 1:length(EntFields)
            EntBlock_all(ne).(EntFields{f}) = mean(tempE.All.(EntFields{f}), 'omitnan');
            EntBlock_early(ne).(EntFields{f})  = mean(tempE.Early.(EntFields{f}), 'omitnan');
            EntBlock_steady(ne).(EntFields{f}) = mean(tempE.Steady.(EntFields{f}), 'omitnan');
            if inclue_rev_flag
                EntBlock_rev(ne).(EntFields{f}) = mean(tempE.afterRev.(EntFields{f}), 'omitnan');
            end
        end
    end

    % reorder struct array into matrix, and take overall means
    EntFields = fieldnames(EntBlock_all);
    for f = 1:length(EntFields)
        SimOutput.All.(EntFields{f}) = mean([EntBlock_all.(EntFields{f})], 'omitnan');
        SimOutput.Early.(EntFields{f}) = mean([EntBlock_early.(EntFields{f})], 'omitnan');
        SimOutput.Steady.(EntFields{f}) = mean([EntBlock_steady.(EntFields{f})], 'omitnan');
        if inclue_rev_flag
            SimOutput.afterRev.(EntFields{f}) = mean([EntBlock_rev.(EntFields{f})], 'omitnan');
        end
    end
    % toc
end

%% Decoder training/testing function

