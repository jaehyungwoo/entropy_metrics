% Figure 2 Illustration of information-theoretic metrics
clearvars; clc

FontS = 15;

beta1 = 10;     % fixed inverse temperature at 10

% initialize simple RL model
SimMod.name = 'RL_basic';      
SimMod.fun  = 'simRL_basic'; % specify simulation function   
SimMod.simpar  =[0.4, 10];       
SimMod.lb      =[ 0   1];   
SimMod.ub      =[ 1 100];   
SimMod.label = "RL1";    
SimMod.plabels = ["\alpha", "\beta"];

% Specify example environment: 80/20, L = 80
rew_probs = [0.8, 0.2];
blockL = 80;
rev_pos = 40;
RewProbs = Construct_reward_schedule(rew_probs, blockL, rev_pos);

% visualize reward probs.
figure(1); clf
set(gcf,'Color','w','Units','normalized','Position',[0, 0, 0.27, 0.2]);
h = plot(RewProbs, '-','LineWidth',2);
    h(1).Color = '#006837';
    h(2).Color = '#F7931E';
xline(rev_pos+.5,'--k','LineWidth',1);
ylim([0 1]);
ylabel("Reward prob.");
xlabel("Trials (exmaple session)");
set(gca,'FontSize',14,'TickDir','out','box','off','LineWidth',1);

%% Simulate choice behavior and compute metrics

numEnv = 100; % number of env. (sim. batch)
numSim = 100; % number of repeated sim per environment
steadyL  = 10;
afterRev = 10;

alpha_set = [0.2, 0.4, 0.6]; % learning rates to simulate
disp(alpha_set);
models = repmat(SimMod,1,length(alpha_set));
for m = 1:length(models)
    models(m).simpar = [alpha_set(m), beta1];
end

% simulate function
[ModOutput] = simulate_entropy_metrics_models(models, RewProbs, numEnv, numSim, steadyL, rev_pos, afterRev);

mod_cols = cool(length(alpha_set));
%% 1. Plot simulated performance (Fig.2b)
figure(11); clf
set(gcf,'Color','w','Units','normalized','Position',[0, 0, 0.27, 0.2]);
hold on; 
for mod = 1:length(models)
    this_dat = reshape([ModOutput{mod}.Trial.pbetter], blockL, numSim)'; 
    shadedErrorBar(1:blockL, mean(this_dat,1), sem(this_dat,1),'lineprops',{'Color',mod_cols(mod,:),'linestyle','-'}); 
end
xline(rev_pos+.5,'--k','LineWidth',1, 'HandleVisibility','off');    
yticks(0:.2:1)
xlabel("Trials"); xlim([1 blockL]);
ylabel("P(Better)"); ylim([0 1]); %yticks
set(gca,'FontSize',FontS,'TickDir','out','box','off','LineWidth',.75);
legend("\alpha = "+alpha_set,'linewidth',.5);

%% 2. Plot time course of metrics (across-block metrics)
% Note trial should start from t=2, since stay/switch is not defined on the first trial

% choose metric to plot
met_set = "H_str"; met_lbl = "H(Str)";          % Fig.2c
% met_set = "n_MIRS"; met_lbl = "n-MIRS";       % Fig.2d
% met_set = "n_MIOS"; met_lbl = "n-MIOS";       % Fig.2e
% met_set = "n_MIROS"; met_lbl = "n-MIROS";     % Fig.2f

% met_set = "ERDS"; met_lbl = met_set;      % etc.
% met_set = "EODS"; met_lbl = met_set;
% met_set = "ERODS"; met_lbl = met_set;

figure(20); clf
set(gcf,'Color','w','Units','normalized','Position',[0, 0, 0.27, 0.2]); hold on
for mod = 1:length(models)
    this_dat = reshape([ModOutput{mod}.Trial.(met_set)], [], numSim)'; 
    shadedErrorBar(2:blockL, mean(this_dat,1,'omitnan'), sem(this_dat,1),'lineprops',{'Color',mod_cols(mod,:),'linestyle','-'}); 
end
xline(rev_pos+.5,'--k','LineWidth',1, 'HandleVisibility','off');    
xlabel("Trials"); xlim([1 blockL]);
ylabel(met_lbl); 
% ylim([0 1]);
set(gca,'FontSize',FontS,'TickDir','out','box','off','LineWidth',0.75);
legend("\alpha = "+alpha_set,'linewidth',.5,'box','off');

%% 3. Within-block metrics (inset)
Block_phase = "All";

figure(21); clf; 
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

% significant test?
disp("-----"+met_set+" comparison:");
[~,Pv1] = ttest2(allDat{1}, allDat{2}); Pv2 = ranksum(allDat{1}, allDat{2}); disp("1. rank sum P = "+Pv2+"; t-test P = "+Pv1);
[~,Pv1] = ttest2(allDat{2}, allDat{3}); Pv2 = ranksum(allDat{2}, allDat{3}); disp("2. rank sum P = "+Pv2+"; t-test P = "+Pv1);
[~,Pv1] = ttest2(allDat{1}, allDat{3}); Pv2 = ranksum(allDat{1}, allDat{3}); disp("3. rank sum P = "+Pv2+"; t-test P = "+Pv1);



