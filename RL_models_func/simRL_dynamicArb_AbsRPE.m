function stats = simRL_dynamicArb_AbsRPE(stats, xpar)
%% simRL_ %%
%PURPOSE:   Simulate player based on q-learning 
%
% INPUT ARGUMENTS
%   stats:  stats of the (simulated) environment thus far
%   xpar: parameters that define the player's strategy
%       xpar(1) = learning rate for choice
%       xpar(2) = inverse temperature
%
% OUTPUT ARGUMENTS
%   stats:  updated with player's probability to choose left for next step
    alphaP     = xpar(1);
    beta_1     = xpar(2);
    alphaM     = xpar(3);
    decayR     = xpar(4);   % decay unchosen
    alphaW     = xpar(5);   % fupdate rate for omega
    
    omega_t     = 0.5;       % initial arbitration weight at trial 1    
    %% Simulate trials
    T = stats.currTrial;
    if stats.currTrial == 1  %if this is the first trial, initialize
        stats.qL(T) = 0.5;      % Q_left 
        stats.qR(T) = 0.5;      % Q_right
        stats.q1(T) = 0.5;      % Q_cir
        stats.q2(T) = 0.5;      % Q_sqr
        
        stats.rpe(T) = NaN;   % RPE = reward prediction error
        stats.omega(T) = omega_t;
        
        stats.p1(T) = 0.5;    % p(choose Option1)
        stats.p2(T) = 0.5;    % p(choose Option1)
        stats.pL(T) = 0.5;    % in this model, L/R = color dimension  
        stats.pR(T) = 0.5;
    else
        % Update action values (reward from previous trial)
        % Update Q_Stimuli
        [stats.q1(T,1), stats.q2(T,1), RPE_Stim] = IncomeUpdateStepRates(stats.r(T-1), stats.c(T-1), stats.q1(T-1),stats.q2(T-1), alphaP,alphaM,decayR);
        % stats.rpe_stim(T-1,1) = RPE_Stim;
        
        % Update Q_Location
        [stats.qL(T,1), stats.qR(T,1), RPE_Loc] = IncomeUpdateStepRates(stats.r(T-1), stats.cloc(T-1), stats.qL(T-1),stats.qR(T-1), alphaP,alphaM,decayR);
        % stats.rpe_loc(T-1,1) = RPE_Loc;
        
        % Update omegaV based on reliability difference: V_chosen as reliability signal
        deltaRel = abs(RPE_Loc) - abs(RPE_Stim);   
                % = V_chosen.Stim - V_chosen.Loc : ranges [-1, 1], positive if Stim more reliable
        if deltaRel > 0
            omega_target = 1;
        else
            omega_target = 0;  
        end
        stats.omega(T,1) = stats.omega(T-1,1) + alphaW*abs(deltaRel)*(omega_target - stats.omega(T-1,1));
    
        % softmax rule for action selection
        % assign sides
        switch stats.stim_on_right(T)
            case -1
                vS_right = stats.q1(T);  % if Option A is on the right
                vS_left  = stats.q2(T);
            case 1
                vS_right = stats.q2(T);  % if Option B is on the right
                vS_left  = stats.q1(T);
        end
        V_left = vS_left*stats.omega(T) + stats.qL(T)*(1 - stats.omega(T));
        V_right = vS_right*stats.omega(T) + stats.qR(T)*(1 - stats.omega(T));
        
        % Decision rule: compute p(Left)
        stats.pL(T,1) = 1 / (1 + exp( -beta_1*(V_left - V_right)));
        stats.pR(T,1) = 1 - stats.pL(T);
        switch stats.stim_on_right(T)
            case -1
                stats.p1(T,1) = stats.pR(T); % Option A is on the right
                stats.p2(T,1) = stats.pL(T);
            case 1
                stats.p2(T,1) = stats.pR(T); % Option B is on the right
                stats.p1(T,1) = stats.pL(T);
        end

    end
end

function [Q_opt1, Q_opt2, rpe] = IncomeUpdateStepRates(reward, choice, Q_opt1, Q_opt2, alphaP, alphaM, decayR)
% updates value estimates Q_i with the decay rate for the unchosen option
    decay_base = 0;     %pars(5);
    
    % disp(choice);
    if choice==1        %chose sqr
        rpe = reward - Q_opt2;
        if reward>0
            Q_opt2 = Q_opt2 + alphaP*(rpe);
        else
            Q_opt2 = Q_opt2 + alphaM*(rpe);
        end
        Q_opt1 = Q_opt1 + decayR*(decay_base-Q_opt1);
    elseif choice==-1   %chose cir
        rpe = reward - Q_opt1;
        if reward>0
            Q_opt1 = Q_opt1 + alphaP*(rpe);
        else
            Q_opt1 = Q_opt1 + alphaM*(rpe);
        end
        Q_opt2 = Q_opt2 + decayR*(decay_base-Q_opt2);
    elseif choice==0
        rpe = nan;
        Q_opt1 = Q_opt1 + decayR*(decay_base-Q_opt1);
        Q_opt2 = Q_opt2 + decayR*(decay_base-Q_opt2);
    else
        error("choice vector error");
    end

end