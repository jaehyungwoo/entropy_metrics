%% subfunc
function [MetOut] = compute_behavioral_and_entropy_met(MetIn, decomp_on, choice, reward, hr_opt, stay, prevR, prevO)
    if decomp_on
        R_decomp = containers.Map({0,1},{'lose','win'});
        O_decomp = containers.Map({0,1},{'worse','better'});
        RO_decomp = containers.Map({0,1,2,3},{'loseworse','losebetter','winworse','winbetter'});    
    else
        R_decomp = [];  % will skip decompositions
        O_decomp = [];
        RO_decomp = [];
    end

    % obtain vectors for computing entropy measures            
    prevRO = binary_to_decimal([prevR, prevO]);
    currBW = choice==hr_opt;
    
    % intialize and compute performance-based metrics
    MetOut = struct;
    MetOut.pwin = mean(reward);
    MetOut.pbetter = mean(currBW);

    % stay decompositions
    MetOut.pStay   = mean(stay);
    MetOut.pStayWin    = mean(stay & prevR) / mean(prevR);
    MetOut.pSwitchLose = mean(~stay & ~prevR) / mean(~prevR);

    % additional info for later normalization of M.I. measures
    MetOut.H_rew = Shannon_Entropy(prevR);
    MetOut.H_opt = Shannon_Entropy(prevO);

    % entropy of strategy
    MetOut.H_str = Shannon_Entropy(stay);    
    MetOut = append_to_fields(MetOut, {Conditional_Entropy_weighted(stay, prevR, "ERDS", R_decomp ), ...                                    
                                    Mutual_Information_decomp(stay, prevR, "MIRS", R_decomp ), ...
                                    Conditional_Entropy_weighted(stay, prevO, "EODS", O_decomp ), ...
                                    Mutual_Information_weighted(stay, prevO, "MIOS", O_decomp), ...
                                    Conditional_Entropy_weighted(stay, prevRO, "ERODS", RO_decomp), ...
                                    Mutual_Information_weighted(stay, prevRO, "MIROS", RO_decomp) });
    MetOut = append_to_fields(MetIn, {MetOut});
end

%
function struct1 = append_to_fields(struct1, struct2s)
    for i = 1:length(struct2s)
        struct2 = struct2s{i};
        for fields = fieldnames(struct2)'
            if isfield(struct1, fields{1})
                struct1.(fields{1}) = [struct1.(fields{1}); struct2.(fields{1})];
            else
                struct1.(fields{1}) = struct2.(fields{1});
            end
        end
    end
end