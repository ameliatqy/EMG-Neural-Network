%EMG_raw = read_hdf;

% % Background Signal
% EMG_background = EMG_raw.trials(16).channels(2).data;
% EMG_background = EMG_background(10000:20000)
% %plot(EMG_background)
% EMG_background_mean = mean(EMG_background)
% EMG_background_std = std(EMG_background)
% threshold = EMG_background_mean + (3*EMG_background_std)

trial_pair = csvread('../trial_num.csv');
trial_pair_size = size(trial_pair);

no_data = 50;

database_features = [];
database_label = [];

train_test_split = 0.8;
split_point = round(train_test_split*trial_pair_size(1));

for i = 1:split_point
    
    cut_start = 0;
    cut_stop = 9;
    
    EMG_time = EMG_raw.trials(trial_pair(i,1)).timeVec;
    EMG_data = EMG_raw.trials(trial_pair(i,1)).channels(2).data;
    
    % Background Signal
    EMG_background = EMG_raw.trials(16).channels(2).data;
    EMG_background = EMG_background(10000:20000);
    %plot(EMG_background)
    EMG_background_mean = mean(EMG_background);
    EMG_background_std = std(EMG_background);
    threshold_default = EMG_background_mean + (3*EMG_background_std);
    
    % Specialization Clean-Up
    if i == 3
        EMG_data = EMG_data(find(EMG_time < 8 & EMG_time > 0.2));
        org_EMG_data = EMG_data;
        threshold_range = find(abs(EMG_data) > threshold);
        EMG_time_cut = EMG_time(find(EMG_time < 8 & EMG_time > 0.2));
    elseif i == 4
        EMG_data = EMG_data(find(EMG_time < 8.5 & EMG_time > cut_start));
        org_EMG_data = EMG_data;
        threshold_range = find(abs(EMG_data) > threshold);
        EMG_time_cut = EMG_time(find(EMG_time < 8.5 & EMG_time > cut_start));
    elseif i == 8
        EMG_data = EMG_data(find(EMG_time < cut_stop & EMG_time > 0.2));
        org_EMG_data = EMG_data;
        threshold_range = find(abs(EMG_data) > threshold);
        EMG_time_cut = EMG_time(find(EMG_time < cut_stop & EMG_time > 0.2));
    elseif i == 14
        EMG_data = EMG_data(find(EMG_time < cut_stop & EMG_time > 0.2));
        org_EMG_data = EMG_data;
        threshold_range = find(abs(EMG_data) > threshold);
        EMG_time_cut = EMG_time(find(EMG_time < cut_stop & EMG_time > 0.2));
        
    elseif i == 16
        EMG_data = EMG_data(find(EMG_time < cut_stop & EMG_time > 0.9));
        org_EMG_data = EMG_data;
        threshold_range = find(abs(EMG_data) > threshold);
        EMG_time_cut = EMG_time(find(EMG_time < cut_stop & EMG_time > 0.9));
    elseif i == 22
        EMG_data = EMG_data(find(EMG_time < 6.8 & EMG_time > cut_start));
        org_EMG_data = EMG_data;
        threshold_range = find(abs(EMG_data) > threshold);
        EMG_time_cut = EMG_time(find(EMG_time < 6.8 & EMG_time > cut_start));
    else
        EMG_data = EMG_data(find(EMG_time < cut_stop & EMG_time > cut_start)); % & EMG_time > 0.5
        org_EMG_data = EMG_data;
        cur_EMG_background_mean = mean(org_EMG_data(find(EMG_time < 1 & EMG_time > 0)));
        cur_EMG_background_std = std(org_EMG_data(find(EMG_time < 1 & EMG_time > 0)));
        cur_threshold_default = cur_EMG_background_mean + (3*cur_EMG_background_std);
    
%         if cur_threshold_default > threshold_default
%             threshold = cur_threshold_default;
%         else
%             threshold = threshold_default;
%         end
        
        threshold = threshold_default;
        threshold_range = find(abs(org_EMG_data) > threshold);
        EMG_time_cut = EMG_time(find(EMG_time < cut_stop & EMG_time > cut_start));
    end
    
    % Curve Smoothing
    EMG_data = detrend(EMG_data);
    EMG_data = abs(EMG_data);
    [b,a]=butter(8,20/1000,'low');
    EMG_data=filtfilt(b,a,EMG_data);
    
    % Get and Threshold Vision Data
    filename = sprintf('../data/data%i.csv', trial_pair(i,2));
    vision_data = csvread(filename);
    vision_data_org = vision_data;
    vision_data = round(vision_data(:,1), 2);
    vision_data_diff = diff(vision_data);
    vision_data_size = size(vision_data);
    
    vision_threshold_range_1 = find(vision_data_diff(1:15) == 0);
    vision_threshold_range_1 = vision_threshold_range_1(end) + 1;
    if i == 21
        vision_threshold_range_2 = find(vision_data_diff(end-10:end) == 0);
    else
        vision_threshold_range_2 = find(vision_data_diff(end-15:end) == 0);
    end
    
    if isempty(vision_threshold_range_2)
        vision_threshold_range_2 = vision_data_size(1);
    else
        vision_threshold_range_2 = (vision_data_size(1)-17) + vision_threshold_range_2(1);
    end
    
    vision_threshold_range_size = size(vision_threshold_range_1:vision_threshold_range_2);
    % Clean up nan
    for n = vision_threshold_range_1:vision_threshold_range_2
        if isnan(double(vision_data(n))) == 1 
            vision_data(n) = [];
        end
    end
    
    % Interpolate
    vision_data = vision_data(vision_threshold_range_1:vision_threshold_range_2);
    xq = linspace(1, size(vision_data,1), no_data);
    vision_data =  interp1(vision_data, xq, 'linear');

    
    vision_data_size_threshold = size(vision_data);
    %vision_data = vision_data_org(vision_threshold_range_1:vision_threshold_range_2);
    sync_idx = round(linspace(threshold_range(1), threshold_range(end), vision_data_size_threshold(2)));
    sync_idx_size = size(sync_idx);
    
    
    
    
    rms_window = 400;
    database_features_trial = [];
    database_label_trial = [];
    
    org_EMG_size = size(org_EMG_data);
    
    for j = 1:sync_idx_size(2)
        if isnan(double(vision_data(j))) == 0 
            if sync_idx(j)+(rms_window/2) > org_EMG_size(2)
                cur_EMG_data = EMG_data(sync_idx(j));
                cur_EMG_data_rms = rms(org_EMG_data((sync_idx(j)-(rms_window/2)):end));
                cur_EMG_data_var = var(org_EMG_data((sync_idx(j)-(rms_window/2)):end));
            elseif sync_idx(j)-(rms_window/2) < 1
                cur_EMG_data = EMG_data(sync_idx(j));
                cur_EMG_data_rms = rms(org_EMG_data(1:(sync_idx(j)+(rms_window/2))));
                cur_EMG_data_var = var(org_EMG_data(1:(sync_idx(j)+(rms_window/2)))); 
            else
                cur_EMG_data = EMG_data(sync_idx(j));
                cur_EMG_data_rms = rms(org_EMG_data((sync_idx(j)-(rms_window/2)):(sync_idx(j)+(rms_window/2))));
                cur_EMG_data_var = var(org_EMG_data((sync_idx(j)-(rms_window/2)):(sync_idx(j)+(rms_window/2))));
            end
        cur_dataentry = double([cur_EMG_data_rms, vision_data(j)]);
        database_features_trial = [database_features_trial cur_dataentry(1:end-1)];
        database_label_trial = [database_label_trial cur_dataentry(end)];
        end
    end
    database_features_trial = (database_features_trial-min(database_features_trial))./(max(database_features_trial)-min(database_features_trial));
    database_label_trial = (database_label_trial-min(database_label_trial))./(max(database_label_trial)-min(database_label_trial));
    
    %database_label_trial = (database_label_trial - deg2rad(40.6))/(deg2rad(107.65) - deg2rad(40.6));
    
    %Reshape
    %database_features_trial = transpose(reshape(database_features_trial, 10, 5));
    %database_label_trial = transpose(reshape(database_label_trial, 10, 5));
    %Reshape
    
    database_features = [database_features; database_features_trial];
    database_label = [database_label; database_label_trial];
    
    % Look at EMG/Vision Data Together
    if isempty(threshold_range) == 0
        subplot(3,1,1)
        plot(EMG_time_cut(threshold_range(1):threshold_range(end)), abs(org_EMG_data(threshold_range(1):threshold_range(end)))/max(abs(org_EMG_data(threshold_range(1):threshold_range(end)))));
        xlabel('Time Elapsed since starting EMG Recording (sec)')
        ylabel('Normalized Amplitude')
        subplot(3,1,2)
        %plot(EMG_data(sync_idx));
        plot(EMG_time_cut(sync_idx), database_features_trial);
        xlabel('Time Elapsed since starting EMG Recording (sec)')
        ylabel('Normalized Amplitude')
        subplot(3,1,3)
        plot(EMG_time_cut(sync_idx), vision_data);
        xlabel('Time Elapsed since starting EMG Recording (sec)')
        ylabel('Normalized Arm Angle')
        filename = sprintf('MacroAnalysis/%i.png', i);
        saveas(gcf, filename);
    end
    
    
    
    
end


dlmwrite('../input_features.csv', database_features)
dlmwrite('../input_label.csv', database_label)

database_features = [];
database_label = [];
database_angle = [];

for i = (split_point+1):trial_pair_size(1)
    
    cut_start = 0;
    cut_stop = 9;
    
    EMG_time = EMG_raw.trials(trial_pair(i,1)).timeVec;
    EMG_data = EMG_raw.trials(trial_pair(i,1)).channels(2).data;
    
    % Background Signal
    EMG_background = EMG_raw.trials(16).channels(2).data;
    EMG_background = EMG_background(10000:20000);
    %plot(EMG_background)
    EMG_background_mean = mean(EMG_background);
    EMG_background_std = std(EMG_background);
    threshold_default = EMG_background_mean + (3*EMG_background_std);
    
    % Specialization Clean-Up
    if i == 3
        EMG_data = EMG_data(find(EMG_time < 8 & EMG_time > 0.2));
        org_EMG_data = EMG_data;
        threshold_range = find(abs(EMG_data) > threshold);
        EMG_time_cut = EMG_time(find(EMG_time < 8 & EMG_time > 0.2));
    elseif i == 4
        EMG_data = EMG_data(find(EMG_time < 8.5 & EMG_time > cut_start));
        org_EMG_data = EMG_data;
        threshold_range = find(abs(EMG_data) > threshold);
        EMG_time_cut = EMG_time(find(EMG_time < 8.5 & EMG_time > cut_start));
    elseif i == 8
        EMG_data = EMG_data(find(EMG_time < cut_stop & EMG_time > 0.2));
        org_EMG_data = EMG_data;
        threshold_range = find(abs(EMG_data) > threshold);
        EMG_time_cut = EMG_time(find(EMG_time < cut_stop & EMG_time > 0.2));
    elseif i == 14
        EMG_data = EMG_data(find(EMG_time < cut_stop & EMG_time > 0.2));
        org_EMG_data = EMG_data;
        threshold_range = find(abs(EMG_data) > threshold);
        EMG_time_cut = EMG_time(find(EMG_time < cut_stop & EMG_time > 0.2));
        
    elseif i == 16
        EMG_data = EMG_data(find(EMG_time < cut_stop & EMG_time > 0.9));
        org_EMG_data = EMG_data;
        threshold_range = find(abs(EMG_data) > threshold);
        EMG_time_cut = EMG_time(find(EMG_time < cut_stop & EMG_time > 0.9));
    elseif i == 22
        EMG_data = EMG_data(find(EMG_time < 6.8 & EMG_time > cut_start));
        org_EMG_data = EMG_data;
        threshold_range = find(abs(EMG_data) > threshold);
        EMG_time_cut = EMG_time(find(EMG_time < 6.8 & EMG_time > cut_start));
    else
        EMG_data = EMG_data(find(EMG_time < cut_stop & EMG_time > cut_start)); % & EMG_time > 0.5
        org_EMG_data = EMG_data;
        cur_EMG_background_mean = mean(org_EMG_data(find(EMG_time < 1 & EMG_time > 0)));
        cur_EMG_background_std = std(org_EMG_data(find(EMG_time < 1 & EMG_time > 0)));
        cur_threshold_default = cur_EMG_background_mean + (3*cur_EMG_background_std);
    
%         if cur_threshold_default > threshold_default
%             threshold = cur_threshold_default;
%         else
%             threshold = threshold_default;
%         end
        
        threshold = threshold_default;
        threshold_range = find(abs(org_EMG_data) > threshold);
        EMG_time_cut = EMG_time(find(EMG_time < cut_stop & EMG_time > cut_start));
    end
    
    % Curve Smoothing
    EMG_data = detrend(EMG_data);
    EMG_data = abs(EMG_data);
    [b,a]=butter(8,20/1000,'low');
    EMG_data=filtfilt(b,a,EMG_data);
    
    % Get and Threshold Vision Data
    filename = sprintf('../data/data%i.csv', trial_pair(i,2));
    vision_data = csvread(filename);
    vision_data_org = vision_data;
    vision_data = round(vision_data(:,1), 2);
    vision_data_diff = diff(vision_data);
    vision_data_size = size(vision_data);
    
    vision_threshold_range_1 = find(vision_data_diff(1:15) == 0);
    vision_threshold_range_1 = vision_threshold_range_1(end) + 1;
    if i == 21
        vision_threshold_range_2 = find(vision_data_diff(end-10:end) == 0);
    else
        vision_threshold_range_2 = find(vision_data_diff(end-15:end) == 0);
    end
    
    if isempty(vision_threshold_range_2)
        vision_threshold_range_2 = vision_data_size(1);
    else
        vision_threshold_range_2 = (vision_data_size(1)-17) + vision_threshold_range_2(1);
    end
    
    vision_threshold_range_size = size(vision_threshold_range_1:vision_threshold_range_2);
    % Clean up nan
    for n = vision_threshold_range_1:vision_threshold_range_2
        if isnan(double(vision_data(n))) == 1 
            vision_data(n) = [];
        end
    end
    
    % Interpolate
    
    vision_data = vision_data(vision_threshold_range_1:vision_threshold_range_2);
    xq = linspace(1, size(vision_data,1), no_data);
    vision_data =  interp1(vision_data, xq, 'linear');

    
    vision_data_size_threshold = size(vision_data);
    %vision_data = vision_data_org(vision_threshold_range_1:vision_threshold_range_2);
    sync_idx = round(linspace(threshold_range(1), threshold_range(end), vision_data_size_threshold(2)));
    sync_idx_size = size(sync_idx);
    
    
    
    
    rms_window = 400;
    database_features_trial = [];
    database_label_trial = [];
    
    org_EMG_size = size(org_EMG_data);
    
    for j = 1:sync_idx_size(2)
        if isnan(double(vision_data(j))) == 0 
            if sync_idx(j)+(rms_window/2) > org_EMG_size(2)
                cur_EMG_data = EMG_data(sync_idx(j));
                cur_EMG_data_rms = rms(org_EMG_data((sync_idx(j)-(rms_window/2)):end));
                cur_EMG_data_var = var(org_EMG_data((sync_idx(j)-(rms_window/2)):end));
            elseif sync_idx(j)-(rms_window/2) < 1
                cur_EMG_data = EMG_data(sync_idx(j));
                cur_EMG_data_rms = rms(org_EMG_data(1:(sync_idx(j)+(rms_window/2))));
                cur_EMG_data_var = var(org_EMG_data(1:(sync_idx(j)+(rms_window/2)))); 
            else
                cur_EMG_data = EMG_data(sync_idx(j));
                cur_EMG_data_rms = rms(org_EMG_data((sync_idx(j)-(rms_window/2)):(sync_idx(j)+(rms_window/2))));
                cur_EMG_data_var = var(org_EMG_data((sync_idx(j)-(rms_window/2)):(sync_idx(j)+(rms_window/2))));
            end
        cur_dataentry = double([cur_EMG_data_rms, vision_data(j)]);
        database_features_trial = [database_features_trial cur_dataentry(1:end-1)];
        database_label_trial = [database_label_trial cur_dataentry(end)];
        end
    end
    database_angle_trial = [min(database_label_trial), max(database_label_trial)];
    database_features_trial = (database_features_trial-min(database_features_trial))./(max(database_features_trial)-min(database_features_trial));
    database_label_trial = (database_label_trial-min(database_label_trial))./(max(database_label_trial)-min(database_label_trial));
    %database_label_trial = (database_label_trial - deg2rad(40.6))/(deg2rad(107.65) - deg2rad(40.6));
    
    
    %Reshape
    %database_features_trial = transpose(reshape(database_features_trial, 10, 5));
    %database_label_trial = transpose(reshape(database_label_trial, 10, 5));
    %Reshape
    
    database_features = [database_features; database_features_trial];
    database_label = [database_label; database_label_trial];
    database_angle = [database_angle; database_angle_trial];
    
    % Look at EMG/Vision Data Together
    if isempty(threshold_range) == 0
        %subplot(4,1,1)
        %plot(EMG_time_cut, abs(org_EMG_data));
        subplot(3,1,1)
        plot(EMG_time_cut(threshold_range(1):threshold_range(end)), abs(org_EMG_data(threshold_range(1):threshold_range(end)))/max(abs(org_EMG_data(threshold_range(1):threshold_range(end)))));
        xlabel('Time Elapsed since starting EMG Recording (sec)')
        ylabel('Normalized Amplitude')
        subplot(3,1,2)
        %plot(EMG_data(sync_idx));
        plot(EMG_time_cut(sync_idx), database_features_trial);
        xlabel('Time Elapsed since starting EMG Recording (sec)')
        ylabel('Normalized Amplitude')
        subplot(3,1,3)
        plot(EMG_time_cut(sync_idx), vision_data);
        xlabel('Time Elapsed since starting EMG Recording (sec)')
        ylabel('Normalized Arm Angle')
        filename = sprintf('MacroAnalysis/%i.png', i);
        saveas(gcf, filename);
    end
    
    
    
    
end


dlmwrite('../test_features.csv', database_features)
dlmwrite('../test_label.csv', database_label)
dlmwrite('../test_angle.csv', database_angle)

%figure(3)
%plot(database_features(:, 1), database_label, '*')