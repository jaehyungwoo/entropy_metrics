function simStats = Simulate_ModelChoice_randR(player, simEnv, choice_dim, randSeed, bait_flag)
% % simulate agent behavior %
%PURPOSE:   Simulate agent behavior on simulated blocks
%           Randomly generate rewards based on assigned prob. of chosen option
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


if ~exist('randSeed','var')
    randSeed = 'shuffle';
end

if ~exist('bait_flag','var')
    bait_flag = false;
end
%% initialize

nT = size(simEnv.rewardprob_AB, 1);   % number of trials

simStats = struct;             % stat for the game

simStats.currTrial = 1;       % starts at trial number 1
simStats.p1 = nan(nT,1);      % probability to choose option 1
simStats.p2 = nan(nT,1);      % probability to choose option 1
simStats.c = nan(nT,1);       % choice vector (stimuli)
simStats.cloc = nan(nT,1);    % choice vector (location)  
simStats.r = nan(nT,1);       % reward vector

simStats.q1 = nan(nT,1);      % action value for *Cir* choice
simStats.q2 = nan(nT,1);      % action value for *Sqr* choice
simStats.qL = nan(nT,1);      % action value for *Left* choice
simStats.qR = nan(nT,1);      % action value for *Right* choice

simStats.rpe = nan(nT,1);     % reward prediction error vector
% simStats.erpe = nan(nT,1);    % expected rpe
% simStats.cL = nan(nT,1);      % choicekernel left
% simStats.cR = nan(nT,1);      % choicekernel right

if bait_flag
    % instantaneous probability (according to baiting rule)
    simStats.instprob = nan(nT,2); 

    % initialize reward prob. with the specified dimension (choice_dim):
    %    AB: stimulus
    %    LR: location 
    simStats.instprob(1,:) = simEnv.("rewardprob_"+choice_dim)(1,:);
end

%% simualte the task
% take the text label for the player, there should be a corresponding Matlab
%function describing how the player will play
simplay1 = str2func(player.label);

simStats.stim_on_right = simEnv.stim_on_right;
consec_unchosen = [0 0];    % consecutive times unchosen (for baiting rule)

rng(randSeed);
if strcmp(choice_dim,'LR')
    % if choice dimension is location-based, use model-predicted choice
    % prob. for location and assign choice to 'cloc'
    for j = 1:nT
        %what is the player's probability for choosing left?
        simStats.currTrial = j;
        simStats = simplay1(simStats, player.params);
        
        % baited rewards: calculate instantaneous probability
        % (Eqn) P_inst = 1 - (1 - P_0)^(n+1)
        simStats.instprob(j,:) = 1 - (1 - simEnv.rewardprob_LR(j,:)) .^ (consec_unchosen + 1);

        % randomly generate choice
        if (rand() < simStats.pL(j))
            simStats.cloc(j) = -1;    % choose left  

            consec_unchosen(1) = 0; % reset 1st option (left)
            consec_unchosen(2) = consec_unchosen(2) + 1; % 2nd option (right) is unchosen
        else
            simStats.cloc(j) = 1;     % choose right

            consec_unchosen(2) = 0; % reset 2nd option
            consec_unchosen(1) = consec_unchosen(1) + 1; % 1st option is unchosen
        end     
        simStats.c(j) = simStats.stim_on_right(j) .* simStats.cloc(j);

        % randomly generate rewards
        chosen_idx = simStats.cloc(j)/2 +1.5; % transforms [-1 1] ==> [1 2]
        if bait_flag
            % baiting: use instantaneous rewrad prob.
            if rand() < simStats.instprob(j,chosen_idx)% P_inst(chosen_idx)
                simStats.r(j) = 1;
            else
                simStats.r(j) = 0;
            end
        else
            % default: use assigned (baseline) reward prob.            
            if (rand() < simEnv.rewardprob_LR(j,chosen_idx))
                simStats.r(j) = 1;
            else
                simStats.r(j) = 0;
            end
        end
    end

    % which 'stimulus' did the player choose?
    % simStats.c = simEnv.stim_on_right .* simStats.cloc;
elseif strcmp(choice_dim,'AB')
    % if choice dimension is stimulus-based, use model-predicted choice
    % prob. for stimulus option and assign choice to 'c'
    for j = 1:nT
        %what is the player's probability for choosing Option A?
        simStats.currTrial = j;
        simStats = simplay1(simStats, player.params);

        % baited rewards: calculate instantaneous probability
        % (Eqn) P_inst = 1 - (1 - P_0)^(n+1)
        simStats.instprob(j,:) = 1 - (1 - simEnv.rewardprob_AB(j,:)) .^ (consec_unchosen + 1);
        
        % randomly generate choice
        % rng(randSeed+j);
        if rand() < simStats.p1(j)
            simStats.c(j) = -1;     % choose Option A  
            consec_unchosen(1) = 0; % reset option A
            consec_unchosen(2) = consec_unchosen(2) + 1; % option B is unchosen
        else
            simStats.c(j) = 1;      % choose Option B
            consec_unchosen(2) = 0; % reset 2nd option (B)
            consec_unchosen(1) = consec_unchosen(1) + 1; % 1st option (A) is unchosen
        end
        simStats.cloc(j) = simStats.stim_on_right(j) .* simStats.c(j);
                
        % randomly generate rewards
        chosen_idx = simStats.c(j)/2 +1.5; % transforms [-1 1] ==> [1 2]
        % rng(randSeed+j*2);
        if bait_flag
            % baiting: use instantaneous rewrad prob.
            if rand() < simStats.instprob(j,chosen_idx)
                simStats.r(j) = 1;
            else
                simStats.r(j) = 0;
            end
        else
            % default: use assigned (baseline) reward prob.
            if rand() < simEnv.rewardprob_AB(j,chosen_idx)
                simStats.r(j) = 1;
            else
                simStats.r(j) = 0;
            end
        end
    end

    % which 'side' did the player choose?
    % simStats.cloc = simEnv.stim_on_right .* simStats.c;
else
    error("Input should be either 'AB' (stimulus) or 'LR' (location) ");
end

% % viz
% figure(2);  clf;
% plot(filtfilt(ones(1,10)/10,1,stats_sim.r),'g'); hold on
% plot(filtfilt(ones(1,10)/10,1,stats_sim.pL),'k');
% plot(filtfilt(ones(1,10)/10,1,double(stats_sim.c==1)),'b');
% plot(filtfilt(ones(1,10)/10,1,double(stats_sim.cloc==-1)),'m');
% legend("P(Win)","P(chooseLeft)","choseSqr","choseLeft",'Location','northoutside');
% revPoint = stats.acq_end + 1;
% plot([revPoint revPoint],[0 1],'--k','HandleVisibility','off');
% xlabel(stats.blockType+", "+stats.HRprob);


end