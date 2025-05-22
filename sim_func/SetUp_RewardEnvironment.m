%% Function: Set up reward environment
function simEnv = SetUp_RewardEnvironment(RewProbs, sd, blockType)
% PURPOSE: creates random simulated block with the assigned reward probs
%          Default: Stimulus-outcome ('What') reward probs, unless specified otherwise
% AUTHORS:   Jae Hyung Woo 20250224
%   
% INPUT:
%       RewProbs        : reward probabilities, [trial x option]
%       sd              : random number seed
%       blockType       : 'What' (stimulus-based), 'Where' (location-based), or 'What2Where' (feature reversal)
% OUTPUT:
%   stats               : struct with the following fields 
%       hr_shape        : higher shape (-1 for Cir, 1 for Sqr)
%       rewardprob      : reward probability matrix for each stimuli (blockL x 2)
%       hr_side         : indicates side where the higher shape appeared on
%       rewardarray     : indicates whether the assigned reward exists in the given stimuli
%                  _AB  : reward array for stim (A vs. B)
%                  _LR  : reward array for location (Left vs. Right)
%       shape_on_right  : indicates which shape is on the right side, assigned pseudorandomly (-1 for A, 1 for B)
%       acq_end         : last trial number in the acquisition phase
%% Create random reward environment 
    rng('shuffle');
    nTrial = length(RewProbs);  % total trial number

    simEnv = struct;       
    simEnv.hr_stim = zeros(nTrial, 1); % better rewarding stimulus
    simEnv.hr_side = zeros(nTrial, 1); % better rewarding location

    RewardArray = nan(nTrial, 2); % will store pre-assigned array of rewards
    
    % assign pseudorandom location for stimuli:
    rng(sd);
    simEnv.stim_on_right = (randi(2,nTrial,1)-1.5)*2;
    opt1_on_right = (simEnv.stim_on_right==-1);

    % number of reversals: when reward schedule changes
    % (for any of the options)
    rev_address = 1 + find(any(diff(RewProbs),2)); % reversal trials
    numBlocks = length(rev_address) + 1;            
        
    % reward array: pre-allocate rewards by block
    % Note: it's also possible to randomly generate reward for each chosen
    % option based on assign schedule, but this method pre-allocates
    % rewards for efficieny
    block_address = [1, rev_address', nTrial+1];    
    for b = 1:numBlocks
        this_trials = block_address(b):block_address(b+1)-1;
        this_N = length(this_trials);        
        this_probs = RewProbs(this_trials,:);
        assert(size(unique(this_probs,'rows'),1)==1, b+": Block has non-unique reward prob.");

        % number of assigned rewards to be assigned for each stim 
        rng(sd*b);
        n_rew1 = [round(this_probs(1,1)*this_N), round((1-this_probs(1,1))*this_N)]; % [# rewards, # no rewards]
        rew1 = [ones(n_rew1(1),1); zeros(n_rew1(2),1)];
        rew1 = rew1(randperm(this_N));

        rng(sd*b);
        n_rew2 = [round(this_probs(1,2)*this_N), round((1-this_probs(1,2))*this_N)];
        rew2 = [ones(n_rew2(1),1); zeros(n_rew2(2),1)];
        rew2 = rew2(randperm(this_N));
        
        RewardArray(this_trials,:) = [rew1, rew2]; 
    end
    simEnv.rewardarray_AB = RewardArray;
    simEnv.rewardarray_LR = RewardArray;

    simEnv.rewardprob_AB = RewProbs;
    simEnv.rewardprob_LR = RewProbs;

    if ~exist('blockType','var')||strcmp(blockType,'What')
        % blockType = 'What';
        % obtain higher reward shape for each trial (What blocks)
        simEnv.hr_stim(RewProbs(:,1) > RewProbs(:,2)) = -1;
        simEnv.hr_stim(RewProbs(:,1) < RewProbs(:,2)) = 1;
        
        % compute higher reward side for each trial
        simEnv.hr_side = simEnv.stim_on_right .* simEnv.hr_stim; % note this works because of [-1, 1] coding
        
        % flip location reward array for 'What' blocks
        simEnv.rewardprob_LR(opt1_on_right,:)  = flip(simEnv.rewardprob_LR(opt1_on_right,:),2); 
        simEnv.rewardarray_LR(opt1_on_right,:) = flip(simEnv.rewardarray_LR(opt1_on_right,:),2);
        
    elseif strcmp(blockType,'Where')
        simEnv.hr_side(RewProbs(:,1) > RewProbs(:,2)) = -1;
        simEnv.hr_side(RewProbs(:,1) < RewProbs(:,2)) = 1;

        % compute higher reward stim for each trial
        simEnv.hr_stim = simEnv.stim_on_right .* simEnv.hr_side;
        
        % flip stim reward array for 'Where' blocks
        simEnv.rewardprob_AB(opt1_on_right,:)  = flip(simEnv.rewardprob_AB(opt1_on_right,:),2);
        simEnv.rewardarray_AB(opt1_on_right,:) = flip(simEnv.rewardarray_AB(opt1_on_right,:),2);
    elseif contains(blockType,'2')
        if strcmp(blockType,'What2Where')
            what_first_flag = 1;
        elseif strcmp(blockType,'Where2What')
            what_first_flag = 0;
        end
        % alternate between 'What' and 'Where' blocks
        for b = 1:numBlocks
            this_trials = false(nTrial,1);
            this_trials(block_address(b):block_address(b+1)-1) = true;
            this_probs = unique(RewProbs(this_trials,:),'rows');
            if mod(b,2)==what_first_flag
                % every odd blocks is 'What'
                irrel_prob = 'rewardprob_LR';
                irrel_array = 'rewardarray_LR';
                rew_dim  = 'hr_stim'; rand_dim = 'hr_side';
            else
                % every even blocks is 'Where'
                irrel_prob = 'rewardprob_AB';
                irrel_array = 'rewardarray_AB';
                rew_dim = 'hr_side'; rand_dim = 'hr_stim';
            end
            simEnv.(rew_dim)(this_trials) = sign(diff(this_probs)); % -1 if first option is better, 1 otherwise
            simEnv.(rand_dim)(this_trials) = simEnv.stim_on_right(this_trials) .* simEnv.(rew_dim)(this_trials);

            % flip reward array for irrelevant dimension
            simEnv.(irrel_prob)(this_trials&opt1_on_right,:)  = flip(RewProbs(this_trials&opt1_on_right,:), 2);
            simEnv.(irrel_array)(this_trials&opt1_on_right,:) = flip(RewardArray(this_trials&opt1_on_right,:), 2);
        end
    elseif strcmp(blockType,'What2Where')
        
    else
        error("Set block type: What or Where");
    end
end