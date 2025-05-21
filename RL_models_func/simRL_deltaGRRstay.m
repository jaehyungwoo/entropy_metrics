function stats = simRL_deltaGRRstay(stats, xpar)
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
    alpha   = xpar(1);
    beta1   = xpar(2); % inv. temp.
    alphaG  = xpar(3);
    beta2   = xpar(4); % stay bias (*GRR)
    
    T = stats.currTrial;
    if stats.currTrial == 1  %if this is the first trial
        stats.q1(T) = 0.5;
        stats.q2(T) = 0.5;
        stats.grr(T) = 0.5;   % initial V
        stats.rpe(T) = NaN;   % RPE = reward prediction error
        stats.deltaGRR(T) = NaN;
        stats.p1(T) = 0.5;    % p(choose Option1) for first trial
        stats.p2(T) = 0.5;    % p(choose Option2)

        stats.stay_bias(T) = 0;
    else
        %% update action values
        if stats.c(T-1)==-1   % if chose Option1 on the previous trial
            stats.rpe(T-1) = stats.r(T-1) - stats.q1(T-1);
            stats.q1(T) = stats.q1(T-1) + alpha*stats.rpe(T-1);
            % if stats.r(T-1) > 0
            %     % Positive RPE: update w/ alpha+
            %     stats.q1(T) = stats.q1(T-1) + alpha*stats.rpe(T-1);
            % else
            %     % Negative RPE: update w/ alpha-
            %     stats.q1(T) = stats.q1(T-1) + alpha*stats.rpe(T-1);
            % end
            stats.q2(T) = stats.q2(T-1);    % unchosen value stays the same
        elseif stats.c(T-1)==1   % else, chose Option2
            stats.rpe(T-1) = stats.r(T-1) - stats.q2(T-1);
            stats.q2(T) = stats.q2(T-1) + alpha*stats.rpe(T-1);
            % if stats.r(T-1) > 0
            %     stats.q2(T) = stats.q2(T-1) + alpha*stats.rpe(T-1);
            % else
            %     stats.q2(T) = stats.q2(T-1) + alpha*stats.rpe(T-1);
            % end
            stats.q1(T) = stats.q1(T-1);    % unchosen value stays the same
        else  %miss trials
            stats.rpe(T-1) = 0;
            stats.q1(T) = stats.q1(T-1);
            stats.q2(T) = stats.q2(T-1);
        end
        
        % update GRR values
        stats.deltaGRR(T-1) = stats.r(T-1) - stats.grr(T-1);
        stats.grr(T,1) = stats.grr(T-1) + alphaG * stats.deltaGRR(T-1);

        %% softmax rule for action selection
        % find p(choose Option1)
        prevChosen = -stats.c(T-1); % A=-1, B=+1; multiply -1 to make it staybias toward Option A (p1)
        stats.stay_bias(T) = beta2 * stats.deltaGRR(T-1) * prevChosen;
        stats.p1(T) = 1 / (1 + exp(-beta1 * (stats.q1(T) - stats.q2(T)) - stats.stay_bias(T)));
        stats.p2(T) = 1 - stats.p1(T);
    end

end
