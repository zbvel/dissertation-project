%Script to analyze the data after lesioning
%change the dim1 and dim2 before running

radius1 = perilesion_radius(1, lesion_corner, 2, 2, lesion_coord);
radius2_total = perilesion_radius(2, lesion_corner, 2, 2, lesion_coord);
radius2 = setdiff(radius2_total, radius1);

%extract the first row from the motor_weights at different stages 
motor_weights_beforelesion_row = motor_weights_beforelesion(1,:);
motor_weights_afterlesion_row = motor_weights_afterlesion(1,:);
motor_weights_100_row = motor_weights_100(1,:);
motor_weights_final_row = motor_weights_final(1,:);

%from the first row extract wts corresponding to the elements within rad1
act1_before = motor_weights_beforelesion_row(radius1);
act1_after = motor_weights_afterlesion_row(radius1);
act1_100 = motor_weights_100_row(radius1);
act1_final = motor_weights_final_row(radius1);

%from the first row extract wts corresponding to the elements within rad2
act2_before = motor_weights_beforelesion_row(radius2);
act2_after = motor_weights_afterlesion_row(radius2);
act2_100 = motor_weights_100_row(radius2);
act2_final = motor_weights_final_row(radius2);

%calculate mean activation for radius1
avg_radius1_before = mean(act1_before)
avg_radius1_after = mean(act1_after)
avg_radius1_100 = mean(act1_100)
avg_radius1_final = mean(act1_final)

%calculate mean activation for radius2
avg_radius2_before = mean(act2_before)
avg_radius2_after = mean(act2_after)
avg_radius2_100 = mean(act2_100)
avg_radius2_final = mean(act2_final)

%find number of elements in radius 1 which are above threshold
num_before_above_threshold_rad1 = size(find(act1_before>0.897))
num_after_above_threshold_rad1 = size(find(act1_after>0.897))
num_100_above_threshold_rad1 = size(find(act1_100>0.897))
num_final_above_threshold_rad1 = size(find(act1_final>0.897))

%find number of elements in radius 2 which are above threshold
num_before_above_threshold_rad2 = size(find(act2_before>0.897))
num_after_above_threshold_rad2 = size(find(act2_after>0.897))
num_100_above_threshold_rad2 = size(find(act2_100>0.897))
num_final_above_threshold_rad2 = size(find(act2_final>0.897))