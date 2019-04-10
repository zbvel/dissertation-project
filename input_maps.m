function PI_input_map = input_maps(n,PIlinear, MIlinear, PtoPI_weights, coordofneighbours1, coordofneighbours2, coordinate_array, coordinate_array4, Pweights, Mweights, Pweights_selected, PIMI_weights_selected, PIMI_weights, Mweights_selected, motor, motor_weights, coordinate_array_wt4)
%returns the matrix with 1 in positions where value of activation in PI or MI
%is greater than the threshold
%have to physically change the element under consideration, like
%Pinput(1,2)=0.5
%If required, also change the representation number from 1 to the
%respective value between 1 and 12
%PI input maps

PI_input_map = zeros(400,1);

PIlinear = PIlinear;
MIlinear = MIlinear;
PtoPI_weights = PtoPI_weights;
Pweights = Pweights;
Mweights = Mweights;
Pweights_selected = Pweights_selected;
Mweights_selected = Mweights_selected;
motor_weights = motor_weights;
motor = motor;
PIMI_weights = PIMI_weights;
PIMI_weights_selected = PIMI_weights_selected;

for y = 1:200
    for x=1:120
    
    Pinput = zeros(2,6);
    Pinput(n) = 0.5;
    
    %propagate to PI
    Pinput_tiled = repmat((Pinput(:))', [400 1]); %400X12
    PIlinear_temp = update_activation(PtoPI_weights, PIlinear, PtoPI_weights, Pinput_tiled, coordofneighbours2, 1, 0.1, -4, 5, 0.8, 0); %400X1
    PIlinear = PIlinear + PIlinear_temp;
    
    %propagate within PI
    Pactivation_selected = PIlinear(coordinate_array);  %400X6 storing the activation of neighbours of each neuron
    PIlinear_temp = update_activation(Pweights, PIlinear, Pweights_selected, Pactivation_selected, coordinate_array, 1, 0.0001, -4, 5, 0.8, 0);
    PIlinear =   PIlinear + PIlinear_temp;
    
    %PI to MI
    Mactivation_selected = MIlinear(coordinate_array);       %retrieves the updated activation values of neighbours        
    PIMI_activation_selected = PIlinear(coordinate_array4);  %retrieves the updated activation values 
    MIlinear_temp = update_activation(PIMI_weights, MIlinear, PIMI_weights_selected, PIMI_activation_selected, coordinate_array4, 1, 0.0001, -2, 3, 0.6, 0);
    MIlinear = MIlinear + MIlinear_temp;
    
    %within MI
    Mactivation_selected = MIlinear(coordinate_array);          
    MItemp = update_activation(Mweights, MIlinear, Mweights_selected, Mactivation_selected, coordinate_array, 1, 0.0001, -2, 3, 0.4, 0);
    MIlinear =  MIlinear +  MItemp;
    
    %MI to lower motor layer
    MIlinear_transpose = MIlinear'; %1X400 row vector
    motor_selected_act = repmat(MIlinear_transpose, [6 1]);  %6X400
    motor_temp = update_activation(motor_weights, motor, motor_weights, motor_selected_act, coordofneighbours1, 2, 0.0001, -2, 1, 0.05, 0);
    motor =  motor + motor_temp;
    
    end

    motor_weights = learning(motor_weights, motor_selected_act, motor, 0, 0.1); %MI to LMN
    PtoPI_weights = learning(PtoPI_weights, Pinput_tiled, PIlinear, 0, 0.2) ;
    PIMI_weights_selected  = learning(PIMI_weights_selected, PIMI_activation_selected, MIlinear, 0, 0.2);
    PIMI_weights_transpose(coordinate_array_wt4) = PIMI_weights_selected; %reset the selected weights in PIMI_weights to new value
end %of learning cycles
indexP = find(PIlinear > 0.3);
PI_input_map(indexP)=1;
PI_input_map = (reshape(PI_input_map, 20, 20))'; %reshapes 400X1 to 20X20

 indexM = find(MIlinear > 0.4);
 MI_input_mapE = MIlinear(indexM)=1;
 reshape


