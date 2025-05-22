function RewProbs = Construct_reward_schedule(rew_probs, blockL, rev_pos)
    block_address = [0, rev_pos, blockL];
    block_address(isnan(block_address)) = [];

    b_idx = cell(1,length(block_address)-1);
    for b = 1:length(block_address)-1
        b_idx{b} = block_address(b)+1:block_address(b+1);
    end
    better1_id = cell2mat(b_idx(1:2:end));
    better2_id = cell2mat(b_idx(2:2:end));

    RewProbs = nan(blockL,2);
    RewProbs(better1_id,1) = rew_probs(1); RewProbs(better2_id,1) = rew_probs(2);
    RewProbs(better2_id,2) = rew_probs(1); RewProbs(better1_id,2) = rew_probs(2);
end