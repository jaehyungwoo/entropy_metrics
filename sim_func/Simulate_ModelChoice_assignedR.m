function simStats = Simulate_ModelChoice_assignedR(player, simEnv, choice_dim, Sd)
% % simulate agent behavior %
%PURPOSE:   Simulate agent behavior on simulated blocks. 
%       Uses pre-assigned rewards on the chosen option for reward feedback
%AUTHORS:   Jae Hyung Woo 20250224
%
%INPUT ARGUMENTS
%   player:    player structure
%       label - the name of the algorithm, the name should correspond to a .m file
%       params - parameters associated with that algorithm
%   simEnv:    info about simulated block environment
%   choice_dim:     choice dimension, specified as either stimulus ('AB') or location ('LR')
%   Sd    :    random seed for generation of choice (softmax rule)
%OUTPUT ARGUMENTS
%   simStats:      simulated choice behavior and latent variables

if nargin<4;    Sd = 'shuffle'; end

%% initialize

nT = size(simEnv.rewardprob_AB,1);   %number of trials

simStats = struct;             % stat for the game

simStats.currTrial = 1;      % starts at trial number 1
simStats.pL = nan(nT,1);      % probability to choose *left* side
simStats.c = nan(nT,1);       % choice vector (stimuli)
simStats.cloc = nan(nT,1);    % choice vector (location)  
simStats.r = nan(nT,1);       % reward vector

simStats.q1 = nan(nT,1);      % action value for *Cir* choice
simStats.q2 = nan(nT,1);      % action value for *Sqr* choice
simStats.qL = nan(nT,1);      % action value for *Left* choice
simStats.qR = nan(nT,1);      % action value for *Right* choice

simStats.cL = nan(nT,1);      % choicekernel left
simStats.cR = nan(nT,1);      % choicekernel right
simStats.erpe = nan(nT,1);    % erpe
simStats.rpe = nan(nT,1);     % reward prediction error vector

%% create reward environment

simStats.stim_on_right = simEnv.stim_on_right;

%% simualte the task

simStats.playerlabel = player.label;
simStats.playerparams = player.params;

% take the text label for the player, there should be a corresponding Matlab
%function describing how the player will play
simplay1 = str2func(player.label);
if strcmp(choice_dim,'LR')
    for j = 1:nT
        %what is the player's probability for choosing left?
        simStats.currTrial = j;
        simStats = simplay1(simStats, simStats.playerparams);
        
        rng(Sd*1000+j); % set random seed
        if (rand() < simStats.pL(j))
            simStats.cloc(j) = -1;    % choose left
            simStats.r(j) = simEnv.rewardarray_LR(j,1); % use preassigned rew.    
        else
            simStats.cloc(j) = 1;     % choose right
            simStats.r(j) = simEnv.rewardarray_LR(j,2);
        end  
        simStats.c(j) = simStats.stim_on_right(j) .* simStats.cloc(j);
    end
    % which 'stimulus' did the player choose?
    % simStats.c = simStats.stim_on_right .* simStats.cloc;
elseif strcmp(choice_dim,'AB')
    for j = 1:nT
        %what is the player's probability for choosing Option A?
        simStats.currTrial = j;
        simStats = simplay1(simStats, simStats.playerparams);
        
        rng(Sd*1000+j); % set random seed
        if (rand() < simStats.p1(j))
            simStats.c(j) = -1;    % choose Option A
            simStats.r(j) = simEnv.rewardarray_AB(j,1); % use preassigned rew.    
        else
            simStats.c(j) = 1;     % choose Option B
            simStats.r(j) = simEnv.rewardarray_AB(j,2);
        end        
        simStats.cloc(j) = simStats.stim_on_right(j) .* simStats.c(j);
    end
    % which 'side' did the player choose?
    % simStats.cloc = simStats.stim_on_right .* simStats.c;
else
    error("Input should be either 'AB' (stimulus) or 'LR' (location) ");
end

end