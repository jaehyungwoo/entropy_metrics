% Simulation 4: Revealing the presence of alternative learning strategies in multidimensional environments
clearvars; %addpath(genpath(pwd));
clc
close all

FontSize = 15;

% set parameters
alphaP = 0.4;
alphaM = 0.4;
beta1  = 10;   % inv. temp.
decayR = 0.2;  % decay unchosen
alphaW = 0.4;  % update rate for omega

%% Initialize models
% 1. basic RL
m = 1;
models(m).name = 'RL_2alpha';      
models(m).fun = 'simRL_2alpha'; % specify simulation function   
models(m).simpar = [alphaP, beta1, alphaM];
models(m).lb     =[ 0   1 0];   % lower bound of params (when fitting)
models(m).ub     =[ 1 100 1];   % upper bound of params
models(m).label  = "Q(Color)-only";    
models(m).plabels = ["\alpha+", "\beta", "\alpha-"];

% 2. Fixed omega
m = 2;    
models(m).name = "RL_dynamicArb_AbsRPE";      
models(m).fun  = "simRL_dynamicArb_AbsRPE"; % specify simulation function   
models(m).simpar = [alphaP, beta1, alphaM, decayR, 0];       
models(m).lb     = [ 0   1 0 0 0];   % lower bound of params (when fitting)
models(m).ub     = [ 1 100 1 1 1];   % upper bound of params
models(m).label = "No arb. (fixed \omega)";    
models(m).plabels = ["\alpha+", "\beta", "\alpha-", "\gamma", "\alpha_{\omega}"];

% 3. Dynamic arbitration 
m = 3;    
models(m).name = "RL_dynamicArb_AbsRPE";      
models(m).fun  = "simRL_dynamicArb_AbsRPE"; % specify simulation function   
models(m).simpar = [alphaP, beta1, alphaM, decayR, alphaW];       
models(m).lb     = [ 0   1 0 0 0 0];   % lower bound of params (when fitting)
models(m).ub     = [ 1 100 1 1 1 1];   % upper bound of params
models(m).label = "Dynamic arb.";    
models(m).plabels = ["\alpha+", "\beta", "\alpha-", "\gamma", "\alpha_{\omega}"];

disp([models.label]);

%% specify environment: 80/20, L = 80
rew_probs = [0.8, 0.2];
blockL = 80;
rev_pos = 40;
RewProbs = Construct_reward_schedule(rew_probs, blockL, rev_pos);

%% Simulate agent behavior
% this uses the model from Woo et al., 2024 bioRxiv
% note 'What' corresponds to color, and 'Where' corresponds to shape

numEnv = 100; % number of env. (sim. batch)
numSim = 100;  % number of repeated sim per environment

Block_set  = ["What", "What2Where"];
Block_lbl  = ["Color\rightarrowColor", "Color\rightarrowShape"];
Block_cols = {'r', 'b'};

decomp_on = 0; % include metric decompositions

disp("Steady state window size: "+steadyL);
ModOutput = cell(1, length(Block_set));
for B = 1:length(Block_set)
    ModOutput{B}.Trial = cell(1,length(models));
    ModOutput{B}.Block.All = cell(1,length(models));
    ModOutput{B}.Block.Steady = cell(1,length(models));
    ModOutput{B}.Block.Unsteady = cell(1,length(models));
    % ModOutput{B}.Block.afterRev = cell(1,length(models));
    disp("Experiment "+B+": "+Block_set(B));
    
    for m = 1:length(models)
        switch class(models)
            case 'cell'
                this_model = models{m};
            case 'struct'
                this_model = models(m);
        end
        player = struct;
        player.label = this_model.fun;
        player.params = this_model.simpar; 
        disp("   > "+m+". "+this_model.label);
    
         % initialize variable
        clear EntTrial EntBlock_all EntBlock_steady EntBlock_unsteady EntBlock_rev simStats   
        EntTrial(numEnv) = struct; % across trials w.r.t. rev    
        EntBlock_all(numEnv) = struct; % within blocks (all trials)
        EntBlock_steady(numEnv) = struct; % within blocks (steady state only)
        EntBlock_unsteady(numEnv) = struct; % within blocks (steady state only)

        % loop through each batch of experiments
        tic
        parfor nE = 1:numEnv
            CompStat = struct;
            CompStat.stay_stim  = []; % stay(1) or switch(0)
            CompStat.stay_loc  = [];
            CompStat.prevO_stim = [];
            CompStat.prevO_loc = [];
            CompStat.prevR = []; % rewarded(1) or not(0)
            CompStat.prevB = [];

            CompStat.Q1 = []; CompStat.Q2 = [];
            CompStat.QL = []; CompStat.QR = [];
            CompStat.omega = [];
    
            % repeat random simulation of each environment   
            tempE = struct;
            for ns = 1:numSim           
                % use pre-assigned reward array (non-baited only)
                simEnv = SetUp_RewardEnvironment(RewProbs, B*numSim*numEnv+(ns+(nE-1)*numSim), Block_set(B));
                simStats = Simulate_ModelChoice_assignedR(player, simEnv, 'AB', B*numSim*numEnv+(ns+(nE-1)*numSim)); 

                stay_stim  = simStats.c(1:end-1)==simStats.c(2:end);
                stay_loc   = simStats.cloc(1:end-1)==simStats.cloc(2:end);
                prevR = simStats.r(1:end-1);
                prevB = simStats.c(1:end-1)==simEnv.hr_stim(1:end-1); 
                prevO_stim = simStats.c(1:end-1)==1; 
                prevO_loc  = simStats.cloc(1:end-1)==1;

                CompStat.stay_stim =  [CompStat.stay_stim; stay_stim']; % [blocks x trials]
                CompStat.stay_loc  =  [CompStat.stay_loc; stay_loc'];
                CompStat.prevO_stim = [CompStat.prevO_stim; prevO_stim'];
                CompStat.prevO_loc  = [CompStat.prevO_loc; prevO_loc'];

                CompStat.prevR = [CompStat.prevR; prevR'];  
                CompStat.prevB = [CompStat.prevB; prevB'];
                CompStat.Q1 = [CompStat.Q1; simStats.q1'];
                CompStat.Q2 = [CompStat.Q2; simStats.q2'];
                CompStat.QL = [CompStat.QL; simStats.qL'];
                CompStat.QR = [CompStat.QR; simStats.qR'];
                if isfield(simStats,'omega')
                    CompStat.omega = [CompStat.omega; simStats.omega'];       
                else
                    CompStat.omega = [CompStat.omega; ones(size(simStats.q1'))];       
                end

                %%% compute whole-block metrics
                [temp_stim] = Omega_behavioral_and_entropy_met("Stim", stay_stim, prevR, prevO_stim, decomp_on);
                [temp_loc]  = Omega_behavioral_and_entropy_met("Loc",  stay_loc, prevR, prevO_loc, decomp_on);
                tempE.All = append_to_fields(temp_stim, {temp_loc});
                tempE.All.pwin = mean(prevR);
                tempE.All.pbetter = mean(prevB);

                %%% steady state: N trials within each phase
                s_idx = [rev_pos-steadyL+1:rev_pos, blockL-steadyL+1:blockL]-1;
                s_idx = s_idx(~isnan(s_idx));
                [temp_stim] = Omega_behavioral_and_entropy_met("Stim", stay_stim(s_idx), prevR(s_idx), prevO_stim(s_idx), decomp_on);
                [temp_loc]  = Omega_behavioral_and_entropy_met("Loc", stay_loc(s_idx), prevR(s_idx), prevO_loc(s_idx), decomp_on);
                tempE.Steady = append_to_fields(temp_stim, {temp_loc});
                tempE.Steady.pwin = mean(prevR(s_idx));
                tempE.Steady.pbetter = mean(prevB(s_idx));

                % early unsteady: N trials within each phase
                s_idx = [(1:unsteadL)+1, rev_pos:rev_pos+unsteadL-1] -1;
                s_idx = s_idx(~isnan(s_idx));
                [temp_stim] = Omega_behavioral_and_entropy_met("Stim", stay_stim(s_idx), prevR(s_idx), prevO_stim(s_idx), decomp_on);
                [temp_loc]  = Omega_behavioral_and_entropy_met("Loc", stay_loc(s_idx), prevR(s_idx), prevO_loc(s_idx), decomp_on);
                tempE.Unsteady = append_to_fields(temp_stim, {temp_loc});
                tempE.Unsteady.pwin = mean(prevR(s_idx));
                tempE.Unsteady.pbetter = mean(prevB(s_idx));
            end
    
            % compute mean values by each batch
            EntFields = fieldnames(tempE.All);
            for f = 1:length(EntFields)
                EntBlock_all(nE).(EntFields{f}) = mean(tempE.All.(EntFields{f}), 'omitnan');
                EntBlock_steady(nE).(EntFields{f}) = mean(tempE.Steady.(EntFields{f}), 'omitnan');
                EntBlock_unsteady(nE).(EntFields{f}) = mean(tempE.Unsteady.(EntFields{f}), 'omitnan');
            end

            %% calculate metric for each trial position
            for t = 1:blockL-1
                stay_stim  = CompStat.stay_stim(:,t);
                stay_loc  = CompStat.stay_loc(:,t);
                prevR = CompStat.prevR(:,t);
                prevB = CompStat.prevB(:,t);
                prevO_stim = CompStat.prevO_stim(:,t);
                prevO_loc = CompStat.prevO_loc(:,t);

                [tempT_stim] = Omega_behavioral_and_entropy_met("Stim", stay_stim, prevR, prevO_stim, decomp_on);
                [tempT_loc] = Omega_behavioral_and_entropy_met("Loc",  stay_loc, prevR, prevO_loc, decomp_on);
                tempEnt = append_to_fields(tempT_stim, {tempT_loc});
                tempEnt.pwin = mean(prevR);
                tempEnt.pbetter = mean(prevB);
                EntFields = fieldnames(tempEnt);
                for f = 1:length(EntFields)
                    EntTrial(nE).(EntFields{f})(t) = tempEnt.(EntFields{f});
                end
            end
            % model values
            EntTrial(nE).omega = mean(CompStat.omega,1,'omitnan'); 
            EntTrial(nE).Q1  = mean(CompStat.Q1,1,'omitnan'); 
            EntTrial(nE).Q2  = mean(CompStat.Q2,1,'omitnan'); 
            EntTrial(nE).QL  = mean(CompStat.QL,1,'omitnan'); 
            EntTrial(nE).QR  = mean(CompStat.QR,1,'omitnan'); 
        end
        ModOutput{B}.Trial{m} = EntTrial;
        ModOutput{B}.Block.All{m} = EntBlock_all;
        ModOutput{B}.Block.Steady{m} = EntBlock_steady;
        ModOutput{B}.Block.Unsteady{m} = EntBlock_unsteady;
        toc
    end
end
disp("Sim complete");

%% Trajectory of performance or omega (show multiple tasks together)
block_lin = ["-","-"];
mod_cols = {ones(1,3)*.65, 'k', 'm'};

% met_set = "pwin"; met_lbl = "P(Win)";
met_set = "pbetter"; met_lbl = "P(Better)";
% met_set = "omega"; met_lbl = "Arbitraion weight \omega";

for B = 1:length(Block_set)
    figure(60+B); clf
    set(gcf,'Color','w','Units','normalized','Position',[0, 0, 0.24, 0.175]); hold on;
    for m = 1:length(models)
        this_dat = assign_output(ModOutput{B}.Trial{m}, met_set, "trial", numEnv);        
        shadedErrorBar(1:size(this_dat,2), mean(this_dat,1,'omitnan'), sem(this_dat), 'lineprops',{'Color',mod_cols{m}, 'linewidth',1.5, 'LineStyle',block_lin(B)});         
    end
    set(gca,'FontSize',FontSize,'TickDir','out','box','off','LineWidth',.75);
    xline(rev_pos+.5,'--k','LineWidth',1, 'HandleVisibility','off');
    ylabel(met_lbl); ylim([0 1]); 
    yticks(0:.2:1);
    xlabel("Trials");
    legend([models.label], 'box','off','location','northwest');
end

%% Plot results: across-trial metrics
mod_cols = {ones(1,3)*.65, 'k', 'm'};
lin_styles = ["-","--"];

B = 1; % choose task (1 or 2)

% met_set = "ERDS" + ["_Stim", "_Loc"]; met_lbl = "ERDS" + ["_{Color}","_{Shape}"]; % Fig.7a-b
met_set = "DeltaERDS"; met_lbl = "ERDS_{Shape} - ERDS_{Color}";     % Fig.7c-d

figure(70+B); clf 
set(gcf,'Color','w','Units','normalized','Position',[0, 0, 0.24, 0.2]); hold on;
for m = 1:length(models)
    for ii = 1:length(met_set)
        if strcmp(met_set,"DeltaERDS")
            this_dat = reshape([ModOutput{B}.Trial{m}.ERDS_Loc] - [ModOutput{B}.Trial{m}.ERDS_Stim], [], numEnv)';
        else
            this_dat = assign_output(ModOutput{B}.Trial{m}, met_set(ii), "trial", numEnv);        
        end
        shadedErrorBar(1:size(this_dat,2), mean(this_dat,1,'omitnan'), sem(this_dat), 'lineprops',{'Color',mod_cols{m}, 'linewidth',1.5, 'LineStyle',lin_styles(ii)});          
    end           
    if prod(unique(sign(mean(this_dat,1,'omitnan'))),'omitnan')==-1; yline(0,":k", 'HandleVisibility','off'); end
end
xline(rev_pos+.5,'--k','LineWidth',1, 'HandleVisibility','off');
ylabel(met_lbl(1));
xlabel("Trials");
set(gca,'FontSize',FontSize,'TickDir','out','box','off','LineWidth',.75);

%% Bar plots of corr. coeff.

met_set = "ERDS" + ["_Stim", "_Loc"]; met_lbl = "ERDS" + ["_{Color}","_{Shape}"]; % Fig.7e-f

Phase_set = ["Unsteady","Steady"];
Phase_col = {'r', [0 0.4470 0.7410]};
phase_alp = [0.5, 0.25];
corrType = 'Pearson';

for B = 1:length(Block_set)
    figure(400+B); clf
    set(gcf,'Color','w','Units','normalized','Position',[0, 0, 0.225, 0.12]);

    bar_dat = nan(length(models), length(Phase_set));
    Pvals   = nan(length(models), length(Phase_set));
    for m = 1:length(models)
        for p = 1:length(Phase_set)
            xdat = [ModOutput{B}.Block.(Phase_set(p)){m}.(met_set(1))];
            ydat = [ModOutput{B}.Block.(Phase_set(p)){m}.(met_set(2))];
            [bar_dat(m,p), Pvals(m,p)] = corr(xdat',ydat','rows','complete','type',corrType);
        end
    end
    Br = bar(1:length(models), bar_dat, 'EdgeColor','none', 'BarWidth', 0.9);
    for p = 1:length(Phase_set)
        Br(p).FaceColor = Phase_col{p};
        Br(p).FaceAlpha = 0.45;
        for m = 1:length(models)
            if Pvals(m,p)<.05
                if Pvals(m,p)<.01
                    if Pvals(m,p)<.001
                        ast = "***";
                    else
                        ast = "**";
                    end
                else
                    ast = "*";
                end                
            else
                ast = "n.s.";                
            end
            text(Br(p).XEndPoints(m), bar_dat(m,p), ast, 'HorizontalAlignment','center','FontSize',20);
        end
    end
    ylim([-0.6 0.2]); yticks(-0.6:.2:.2);
    ylabel("corr. coeff.");
    xticklabels([models.label]);
    set(gca,'FontSize',FontSize,'TickDir','out','box','off','LineWidth',.75);
end

%% subfunc
function [tempMet] = Omega_behavioral_and_entropy_met(label, stay, prevR, prevO, decomp_on)
    label = "_"+label;
    if decomp_on
        R_decomp = containers.Map({0,1},{'lose','win'});    
        O_decomp = containers.Map({0,1},{'opt1','opt2'});
        RO_decomp = containers.Map({0,1,2,3},{'loseworse','losebetter','winworse','winbetter'});
    else
        R_decomp = [];
        O_decomp = [];
        RO_decomp = [];
    end
    % obtain vectors for computing entropy measures            
    prevRO  = binary_to_decimal([prevR, prevO]);

    % intialize and compute performance-based metrics
    tempMet = struct;

    % stay decompositions
    tempMet.("pStay"+label)   = mean(stay);
    tempMet.("pStayWin"+label)    = mean(stay & prevR) / mean(prevR);
    tempMet.("pStayLose"+label)   = mean(stay & ~prevR) / mean(~prevR);
    % tempMet.pSwitchWin  = mean(~stay & prevR) / mean(prevR);
    % tempMet.pSwitchLose = mean(~stay & ~prevR) / mean(~prevR);

    % additional info for later normalization of M.I. measures
    % tempMet.H_rew = Shannon_Entropy(prevR);
    % tempMet.H_opt = Shannon_Entropy(prevO);

    % entropy of strategy
    tempMet.("H_str"+label) = Shannon_Entropy(stay);    
    tempMet = append_to_fields(tempMet, {Conditional_Entropy_decomp(stay, prevR, "ERDS"+label, R_decomp ), ...                                    
                                    Mutual_Information_decomp(stay, prevR, "MIRS"+label, R_decomp ), ...                                    
                                    Conditional_Entropy_decomp(stay, prevO, "EODS"+label, O_decomp ), ...
                                    Mutual_Information_decomp(stay, prevO, "MIOS"+label, O_decomp), ...
                                    Conditional_Entropy_decomp(stay, prevRO, "ERODS"+label, RO_decomp), ...
                                    Mutual_Information_decomp(stay, prevRO, "MIROS"+label, RO_decomp) });
end
