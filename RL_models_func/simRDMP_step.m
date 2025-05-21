function stats = simRDMP_step(stats, xpar)
%% simRDMP %%
%PURPOSE:   Simulate player based on reward-dependent metaplasticity
%
% INPUT ARGUMENTS
%   stats:  stats of the (simulated) environment thus far
%   xpar: parameters that define the player's strategy
%       xpar(1) = baseline learning rate (q1) for choice
%       xpar(2) = inverse temperature
%       xpar(3) = baseline metaplastic transition rate (p1)    
% OUTPUT ARGUMENTS
%   stats:  updated with player's probability to choose left for next step

%% assign parameters  
alpha_1 = xpar(1);   % transition base prob, alpha_1 (or q1)
beta1   = xpar(2);   % inv. temp      
mp_1    = xpar(3);   % meta-transition base prob, p1 

N_meta  = 4;         % # of meta-states

% Potentiation
PotentMAT = struct; 
qArray = [zeros(1,N_meta), alpha_1 .^ ( ( (N_meta-2)*(1:N_meta) + 1 ) / (N_meta-1)) ];
tempQ  = [zeros(N_meta-1,N_meta*2); qArray; zeros(N_meta,N_meta*2)];
PotentMAT.Q = tempQ - diag(sum(tempQ));
pArray = [ mp_1.^(N_meta-1:-1:1), 0, mp_1.^(1:N_meta-1) ];
tempP  = diag(pArray,1);
PotentMAT.P = tempP - diag(sum(tempP));

% Depression
DepressMAT = struct;
DepressMAT.Q = eye(N_meta*2) + tempQ - diag(sum(tempQ));
DepressMAT.Q = DepressMAT.Q(N_meta*2:-1:1, N_meta*2:-1:1) - eye(N_meta*2);
DepressMAT.P = eye(N_meta*2) + tempP - diag(sum(tempP));
DepressMAT.P = DepressMAT.P(N_meta*2:-1:1, N_meta*2:-1:1) - eye(N_meta*2);

T = stats.currTrial;
if T==1  
    % if this is the first trial
    % set initial synapses at 0.5 (equal proportion across meta-states)
    % syanpse index order: strong,deep-->strong,m=1-->weak,m=1-->weak,deep
    initial_syn = [0.5/N_meta*ones(N_meta,1); 0.5/N_meta*ones(N_meta,1)]';
    stats.syn_1(T,:) = initial_syn;
    stats.syn_2(T,:) = initial_syn;
    stats.syn_L(T,:) = initial_syn;
    stats.syn_R(T,:) = initial_syn;
    
    % strong synapses sums up to Q-value
    stats.qL(T) = 0.5;      % V_left 
    stats.qR(T) = 0.5;      % V_right
    stats.q1(T) = 0.5;      % V_cir
    stats.q2(T) = 0.5;      % V_sqr
    
    stats.rpe = NaN;

    stats.pL(T) = 0.5;      % p(Left)
    stats.p1(T) = 0.5;      % p(OptionA)
else
    %% Update action values (use reward from previous trial)
    % Update V_Stimuli
    [Syn1, Syn2, stats.rpe(T)] = RDMP_ReturnUpdateStepbyStep(stats.r(T-1),stats.c(T-1),stats.syn_1(T-1,:)',stats.syn_2(T-1,:)', PotentMAT, DepressMAT);
    stats.syn_1(T,:) = Syn1';
    stats.syn_2(T,:) = Syn2';

    %% softmax rule for action selection
    % assign location
    stats.q1(T) = sum(stats.syn_1(T,N_meta:-1:1));
    stats.q2(T) = sum(stats.syn_2(T,N_meta:-1:1));
    switch stats.stim_on_right(T)
        case -1
            vS_right = stats.q1(T);  % Circle on the right
            vS_left = stats.q2(T);
        case 1
            vS_right = stats.q2(T);
            vS_left = stats.q1(T);
    end
    DV_left  = vS_left;
    DV_right = vS_right;
        
    % compute p(Left)
    stats.pL(T) = 1/(1+exp(-beta1*(DV_left-DV_right)));

    % compute p(OptionA)
    stats.p1(T) = 1/(1+exp(-beta1*(stats.q1(T)-stats.q2(T))));
    stats.p2(T) = 1 - stats.p1(T);
end
end

%% update synapses
function [V_syn1,V_syn2,rpe] = RDMP_ReturnUpdateStepbyStep(reward,choice,V_syn1,V_syn2,PotentMAT,DepressMAT)

    if choice==1        %chose Option2 (Sqr/Right)
        rpe = reward - sum(V_syn2(1:length(V_syn2)/2));
        if reward>0
            % potentiate V2
            V_syn2 = V_syn2 + PotentMAT.Q*(V_syn2);     % plastic transition first
            V_syn2 = V_syn2 + PotentMAT.P*(V_syn2);     % then meta-transition
        else
            % depress V2
            V_syn2 = V_syn2 + DepressMAT.Q*(V_syn2);
            V_syn2 = V_syn2 + DepressMAT.P*(V_syn2);
        end
    elseif choice==-1   %chose Option1 (Cir)
        rpe = reward - sum(V_syn1(1:length(V_syn1)/2));
        if reward>0
            % potentiate V1
            V_syn1 = V_syn1 + PotentMAT.Q*(V_syn1);
            V_syn1 = V_syn1 + PotentMAT.P*(V_syn1);
        else
            % depotentiate V1
            V_syn1 = V_syn1 + DepressMAT.Q*(V_syn1);
            V_syn1 = V_syn1 + DepressMAT.P*(V_syn1);
        end
    elseif choice==0
        rpe = NaN;
    else
        error('choice vector error');
    end

end