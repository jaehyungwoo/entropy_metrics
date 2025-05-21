function [negloglike, nlls, choiceProb, V_hist] = funRL_2alpha_fixedBeta(xpar, dat)
% % funRL_basic % 
%PURPOSE:   Function for maximum likelihood estimation, called by fit_fun().
%
%INPUT ARGUMENTS
%   xpar:       alpha, beta
%   dat:        data
%               dat(:,1) = choice vector
%               dat(:,2) = reward vector
%
%OUTPUT ARGUMENTS
%   negloglike:      the total sum of negative log-likelihood to be minimized
%   nlls      :      negative log-likelihood by each trial
%   V_hist    :      struct for storing history of value estimates of two options, V(A) and V(B)
%   choiceProb:      model-predicted probability of choosing option 1

%% 
alpha_plus  = xpar(1);    % positive learning rate
beta        = dat{3};    % pass fixed inverse temperature w/o fitting
alpha_minus = xpar(2);    % negative learning rate

nt = size(dat{1},1);
negloglike = 0;

v_1 = 0.5;  % value functions for two options, initialized at zero
v_2 = 0.5;

V_hist.v_1 = nan(nt,1);   % V_cir
V_hist.v_2 = nan(nt,1);   % V_sqr
V_hist.rpe = nan(nt,1);   % Reward Prediction Error (RPE)

choice_stim = dat{1}; % chosen stimulus vector
reward_vec  = dat{2};

choiceProb = zeros(1,nt);
nlls = zeros(1,nt);

for k = 1:nt
%% Loop through trials
    % track record of all V's
    V_hist.v_1(k) = v_1;
    V_hist.v_2(k) = v_2;
    
    % obtain final choice probabilities for two competing options
    p_2 = exp(beta*v_2)/(exp(beta*v_2)+exp(beta*v_1));
    p_1 = 1 - p_2;
    if p_2==0
        p_2 = realmin;   % Smallest positive normalized floating point number, because otherwise log(zero) is -Inf
    end
    if p_1==0
        p_1 = realmin;
    end
    choiceProb(k) = p_1;
    
    % compare with actual choice to calculate log-likelihood
    choice = choice_stim(k);
    if choice==1
        logp = log(p_2);
    elseif choice==-1
        logp = log(p_1);
    else
        logp = 0;
    end
    nlls(k) = -logp;    % for this trial only
    negloglike = negloglike - logp; % cumulative sum
    
    % Learning: update value for the performed action / chosen options
    reward = reward_vec(k);
    if choice==-1   %chose the first option (coded by -1)
        RPE = reward - v_1;
        if reward>0
            v_1 = v_1 + alpha_plus*(RPE);   % if rewarded (when RPE>0)
        else
            v_1 = v_1 + alpha_minus*(RPE);  % if unrewarded (when RPE<0)
        end
    elseif choice==1        %chose the second option (coded by +1)
        RPE = reward - v_2;
        if reward>0
            v_2 = v_2 + alpha_plus*(RPE);
        else
            v_2 = v_2 + alpha_minus*(RPE);
        end
    end
    V_hist.rpe(k) = RPE;

end

end