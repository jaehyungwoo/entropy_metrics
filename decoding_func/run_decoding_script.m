% script for decoding positivity bias

DecodeVar_set = {["n_MIRS_lose"], ["all"]}; 
DecodeVar_lbl = ["n-MIRS-", "All metrics"];

% take first 10 trials only; no need to include reversals
blockL = 11;
windowL = 10; 
rev_pos = NaN;
RewProbs = Construct_reward_schedule(rew_probs, blockL, rev_pos);

%% simulation settings
sampleN = 10; % # of experiments
agentN  = 20; % # of agents in each group
withinN = 10; % # of within-parameter samples to take average
radi  = 0.15;  % circle radius (range around base alpha's)
sigma = 0.03;  % Gaussian std

a_min = 0 + radi; 
a_max = 1 - radi; % min/max of alpha centers

%% Run decoding on the sample agents
disp("Q1 = "+Q1);
disp("agentN = "+agentN+", radius = "+radi+", sigma = "+sigma);

% block info
disp(RewProbs(1,:));
disp(size(RewProbs));

% don't run if file exists
fname = "AccuMAT_"+agentN+"agents_r"+radi+"_sig"+sigma+"_samp"+sampleN+"x"+withinN+"_P"+rew_probs(1)+"_P"+rew_probs(2)+"_L"+blockL+"_"+trial_port+windowL+".mat";
disp(fname);

DecodeOpt = struct; % structure containing decoding options
DecodeOpt.VarSet        = DecodeVar_set; % set of used features (variables)
DecodeOpt.VarLbl        = DecodeVar_lbl;
DecodeOpt.trial_portion = "first";    % compute metrics from first N trials of the block
DecodeOpt.windowL       = windowL;
DecodeOpt.agentN        = agentN;
DecodeOpt.sampleN       = sampleN;
DecodeOpt.withinN       = withinN;
DecodeOpt.radius        = radi;   % radius of the agent boundary
DecodeOpt.sigma         = sigma;    

% repeat experiments N times
PosNegative_KFold = cell(sampleN, 1);
PosNegative_LOO   = cell(sampleN, 1);
parfor Sd = 1:sampleN    
    % draw neutral agent
    rng(Sd);    

    % draw optimistic agent; specify triangle where the center lies
    rng(Sd);
    t1 = [a_min a_min+radi]; %t1 = [a_min a_min+.1]; 
    t2 = [a_min a_max]; 
    t3 = [a_max-radi a_max]; %t3 = [a_max-.1 a_max];
    u = rand(agentN, 1); v = rand(agentN, 1);
    idx = u + v > 1;
    u(idx) = 1 - u(idx);
    v(idx) = 1 - v(idx);
    opt_alphas = (1 - u - v) * t1 + u * t2 + v * t3; % (alpha-, alpha+)

    % run one instance
    sStart = tic;
    [PosNeut, PosNeg, AllGroupDat] = sample_and_decode_positivity(RewProbs, DecodeOpt, opt_alphas, beta1);    
  
    % assign data
    PosNegative_KFold{Sd} = PosNeg.Kfold;
    PosNegative_LOO{Sd}   = PosNeg.LOO';
    disp("------------[Seed "+Sd+"] "+toc(sStart)/60+" minutes elapsed");
end

% Save output
PosNegative.Kfold = PosNegative_KFold;
PosNegative.LOO   = PosNegative_LOO;

save("output/decoding/"+fname, 'PosNegative','DecodeOpt');
disp("File saved: output/decoding/"+fname);

all_end = toc(all_start);
disp("Total elapsed time is "+toc(all_start)/60+" minutes");