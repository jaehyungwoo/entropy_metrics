function stats = simRL_basic(stats, xpar)
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
    beta    = xpar(2);
    
    T = stats.currTrial;
    if stats.currTrial == 1  %if this is the first trial
        stats.q1(T) = 0.5;
        stats.q2(T) = 0.5;
        stats.rpe(T) = NaN;   % RPE = reward prediction error
        stats.p1(T) = 0.5;    % p(choose Option1)
        stats.p2(T) = 0.5;    % p(choose Option1)
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
        
        %% softmax rule for action selection
        % find p(choose Option1)
        stats.p1(T) = 1 / (1 + exp(-beta * (stats.q1(T) - stats.q2(T))));
        stats.p2(T) = 1 - stats.p1(T);
    end

end
