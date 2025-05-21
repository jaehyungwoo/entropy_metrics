function output = Conditional_Entropy_decomp(Y, X, metric_name, decomp_map)
%PURPOSE:   Compute conditional entropy H(Y|X) from two logical vectors 
%AUTHORS:   Jae Hyung Woo 04/10/2025; adapted from codes by Ethan Trepka 10/05/2020
%
%INPUT ARGUMENTS
%   Y:  vector of conditioned variable, each uq value is an event; e.g., 1 = stay, 0 = switch
%   X:  vector of conditioning variable, each value is an event; e.g., 1 = win, 0 = lose 
%   metric_name: string, name of conditional entropy metric
%   decomp_map: map from "events" in x to strings to label to decompositions that corresponds with the label, 
%               e.g., {1,0 -> "win","lose"}
% 
%OUTPUT ARGUMENTS
%   output: 
%       (metric_name) : equivalent to H(Y|X)
%       (metric_name + decomp_x) : decompositions which sum up to whole metric; equivalent to P(x) * H(Y|X = x)
%       (u+ metric_name + decomp_x) : (unweigthed) conditional entropy for specific instance, equal to H(Y|X = x)
    
    assert(length(X)==length(Y),"Input vectors should have same lengths");
    
    % exclude NaN entries if any
    if sum(isnan(Y))>0||sum(isnan(X))>0
        nanIDX = isnan(Y)|isnan(X);
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
        for i = 1:decomp_map.Count
            output.(metric_name+"_"+decomp_vals(i)) = NaN;
            output.("u"+metric_name+"_"+decomp_vals(i)) = NaN;
        end
    end  
    %% sum through all cases    
    H_Y_given_X = nan(1, length(x_unique)); % intialize whole metric decompositions (vector)
    prob_X = nan(1, length(x_unique));      % vector containing prob. for all cases of 'X'

    % loop through each instance of X; e.g., {win, lose}
    for x_num = 1:length(x_unique)
        H_Y_given_x = nan(1,length(y_unique)); % initialize conditional entropy bins (vector) for specfic instances of X; e.g., H(Str|R=win)
                                               
        x_signal = (X==x_unique(x_num)); 
        prob_x = mean(x_signal);  % e.g., p(win)        
        prob_X(x_num) = prob_x;   % save this value for later summing                       
        
        % loop through each instance of Y in {stay, switch}
        for y_num = 1:length(y_unique)
            y_signal = (Y==y_unique(y_num));
            prob_y_given_x = mean(y_signal & x_signal ) / prob_x; % e.g., p(stay|win) = p(stay,win) / p(win)

            H_Y_given_x(y_num) = - prob_y_given_x * log2(prob_y_given_x);  % e.g., H(Str=stay|R=win) = p(stay|win) * log2[p(stay|win)]
            if prob_y_given_x==0
                H_Y_given_x(y_num) = 0; % correct multiplication with zero as zero
            end
        end
        H_Y_given_X(x_num) = nan_sum(H_Y_given_x); 

        if decompose_flag
            output.("u"+metric_name+"_"+decomp_map(x_unique(x_num))) = nan_sum(H_Y_given_x); % conditional entropy decompsition; e.g., H(Str|R=Win)
            output.(metric_name+"_"+decomp_map(x_unique(x_num))) =  prob_x * H_Y_given_X(x_num); % weighted by 'X' event; e.g., p(win)*H(Str|R=win)
        end
    end
    output.(metric_name) = nan_sum(prob_X .* H_Y_given_X); % whole metric sum, H(Y|X)
                             % e.g., H(Str|R) = p(win)*H(Str|R=win) + p(lose)*H(Str|R=lose)
end

function Sum = nan_sum(Array)
    % avoid returning zero when all elements are NaN
    if all(isnan(Array))
        Sum = NaN;
    else
        Sum = sum(Array, 'omitnan');    
    end
end
