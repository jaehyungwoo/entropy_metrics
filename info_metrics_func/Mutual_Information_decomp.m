function output = Mutual_Information_decomp(Y, X, metric_name, decomp_map)
%PURPOSE:   Compute mutual information I(X;Y) and its decompositions from two logical vectors 
%AUTHORS:   Jae Hyung Woo 04/10/2025; adapted from codes by Ethan Trepka 10/05/2020
%
%INPUT ARGUMENTS
%   Y:  vector of conditioned variable, each uq value is an event; e.g., 1 = stay, 0 = switch
%   X:  vector of conditioning variable, each value is an event; e.g., 1 = win, 0 = lose 
%   metric_name: string, name of conditional entropy metric
%   decomp_map: map from "events" in x to strings to label to decompositions that corresponds with the label, 
%               e.g., {1,0 -> "win","lose"}
%OUTPUT ARGUMENTS
%   output: 
%       (metric_name) : equivalent to I(X;Y)
%       (metric_name + decomp_x) : decompositions which sum up to whole metric; equivalent to P(x) * I(Y; X = x)
%       (n_ + metric_name) : normalized M.I. metric, simply divided by H(Y)

    assert(length(X)==length(Y));        
    if sum(isnan(Y))>0||sum(isnan(X))>0
        nanIDX = isnan(Y)|isnan(X); % exclude NaN entries if any
        X(nanIDX) = [];
        Y(nanIDX) = [];
    end    
    y_unique = unique(Y);
    x_unique = unique(X);

    if ~exist('decomp_map', 'var')||isempty(decomp_map)
        decompose_flag = false;
    else
        decompose_flag = true;
    end
    
    if decompose_flag
        decomp_vals = values(decomp_map);
        output.(metric_name) = NaN;
        for i = 1:decomp_map.Count
            output.(metric_name+"_"+decomp_vals(i)) = NaN;
        end
    end  
    %% sum through all cases 
    weighted_sum = nan(1, length(x_unique));
    I_Y_given_X  = nan(1, length(x_unique)); % intialize whole metric decompositions (vector)
    H_Y = Shannon_Entropy(Y); % total entropy, H(Y)

    % loop through each instance of X; e.g., {win, lose}
    for x_num = 1:length(x_unique)
        H_Y_given_x = nan(1,length(y_unique)); % initialize conditional entropy bins (vector) for specfic instances of X; e.g., H(Str|R=win)
        x_signal = (X==x_unique(x_num));
        prob_x = mean(x_signal);  % e.g., p(win)

        % loop through all instances of Y in {stay, switch}
        weighted_sum_vec = nan(1, length(y_unique)); 
        for y_num = 1:length(y_unique)
            y_signal = (Y==y_unique(y_num));
            prob_y = mean(y_signal);
            prob_x_and_y = mean(y_signal & x_signal);

            weighted_sum_vec(y_num) = prob_x_and_y * log2( prob_x_and_y / (prob_x * prob_y) );
            
            prob_y_given_x = prob_x_and_y / prob_x; % e.g., p(stay|win) = p(stay,win) / p(win)
            H_Y_given_x(y_num) = - prob_y_given_x * log2(prob_y_given_x);  % e.g., H(Str=stay|R=win) = p(stay|win) * log2[p(stay|win)]
            if prob_y_given_x==0
                H_Y_given_x(y_num) = 0; % correct multiplication with zero as zero
            end
        end        
        I_Y_given_X(x_num) = H_Y - nan_sum(H_Y_given_x); % computed as I(X=x;Y) = H(Y) - H(Y|X=x)
        weighted_sum(x_num) = nan_sum(weighted_sum_vec);
        if decompose_flag
            output.("p"+metric_name+"_"+decomp_map(x_unique(x_num))) = prob_x * I_Y_given_X(x_num);                 
        end
    end    
    output.(metric_name) = nan_sum(weighted_sum); 

    % Normalized metrics
    metric_fields = fieldnames(output);
    for m = 1:length(metric_fields)
        output.("n_"+metric_fields{m}) = output.(metric_fields{m}) / H_Y; % normalized by H(Y); coefficients of constraints, C_XY
    end
end

function Sum = nan_sum(Array)
    % avoid returning zero when all elements are NaN
    if all(isnan(Array))
        Sum = NaN;
    else
        Sum = sum(Array, 'omitnan');    
    end
end