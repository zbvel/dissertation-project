function MI_output_map = printMIoutput(motor_weights, muscle, threshold) 
%PRINTING OF MI output map 
%tuned to length or tension of a particular muscle
%measures the weights above threshold from MI to LMN, pertaining to the
%length or tension of the muscle
%muscle stands for the numerical value of the muscle, i.e 1,2,3,4,5,6 for
%E,F,AB,AD,O,C, the same order as in LMN layer
% motor_weights is 6X400, so find all elements in row of Extensor with value above threshold 0.4


element_zero = zeros(6,400);  %same size as motor_weights
element_dash=zeros(1,400);
MI_op_element = find(motor_weights(muscle,:)>threshold); %find values above the threshold in specific row
element_dash(MI_op_element) = muscle;  %prints the numerical value of the 
%muscle in place of weights greater than the threshold, others remain 0
MI_output_map = (reshape(element_dash, 20, 20))'; %reshapes 400X1 to 20X20,


%------------------------------------------------------------------------------------------------


