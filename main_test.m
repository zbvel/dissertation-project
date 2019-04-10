%---------Set the number of elements---------------------------------------
s=20; %'s' is for single and is used in place of 20 in the final code
d=400; %'d' is for double and is used in place of 400 in final code


%----------The MI layer - has 20X20 elements - declarations----------------
%%% WITHIN PI
MI = ones(s,s);  %the initial random matrix for activation values in MI layer
MI=MI * 0.03; %set activation of all elements to 0.03
MIlinear_temp = zeros(d,1);
input_activation = zeros(s,s); %this will be used to provide input activation patch to MI layer
Mweights = 0.1 + (1-0.1)*rand(d,d); %the intra-layer weights matrix
Mweights_selected = zeros(d,6); %to store the neighbour weight values for each element
Mactivation_selected = zeros(d,6); %to store the activations of 6 neighbours 
coordinate_array = zeros(d,6); %to store the retrieved coordinates (i,j) of neighbours
coordinate_array_wt1 = zeros(d,6); %stores the linearised wt-coordinate values from 1-160000
coordinate_array_wt4 = zeros(d,61); 
coordinate_array4 = zeros(d,61);

% PI to MI

%PIMI_weights_zeros = zeros(d,d);  % create a zeros matrix of same size as weights matrix 400X400
PIMI_weights = 0.1 + (1-0.1)*rand(d,d); %the PI to MI inter-layer weights matrix - already declared once before, remove
PIMI_weights_selected = zeros(d,61); %to store the neighbour weight values for each element
PIMI_activation_selected = zeros(d,61); %to store the activations of 61 neighbours 

%-----------Lower Motor Neuron Layer - declarations-------------------------
%%%
%Lower Motor Neuron Layer or 'motor' - has 6 elements storing the
%activation received by each of the 6 muscles from MI layer

motor = ones(6,1) ;   %stores the activation values of 6 muscles in this order; E,F,AB,AD,O,C
motor = motor * 0.03;
motor_weights = 0.1 + (1-0.1)*rand(6,d);   %stores the inter-layer weights from MI layer to motor

%----------Proprioceptive Input Layer - declarations-----------------------
%%%
% The Proprioceptive Input layer - has 12 elements (2X6) storing the length
% and tension values for each of the 6 muscles

Pinput = ones(2,6); %row1 stores length and row2 stores tension of each of the 6 muscles in above order
Pinput = Pinput * 0.03;
%following is only for printing PI input map for extensor length and tension after training
%Pinput(1,1)=0.5; %extensor length is set to 0.5
%Pinput(2,1)=0.5; %extensor tension is set to 0.5

PtoPI_weights = 0.1 + (1-0.1)*rand(d,12);  %stores inter-layer weights from Pinput to PI; 400X12

%----------PI Layer has 20X20 elements - declarations------------------------------
%%%
PI = ones(s,s);              %activation values stored here; 20X20
PI = PI * 0.03;
PIlinear_temp = zeros(d,1);
Pweights = 0.1 + (1-0.1)*rand(d,d);        %stores intra-layer weights; 400X400
Pweights_selected = zeros(d,6); %selected intra-layer weights
Pactivation_selected = zeros(d,6); %to store the activations of 6 neighbours 

%-----------Input patch -----------------------------------------------------------
neighbours7 = zeros(1,7);
%----------------------------------------------------------------------------------

for x=1:s
    for y=1:s
        IJ = Calculate_IJarray1(x,y);
        IJ4 = Calculate_IJarray4(x,y);
        coordinate = (x-1)*s + y;
        coordinate_array_wt1(coordinate,:) = (coordinate-1)*d + IJ; %dX6, to retrieve selected wts from 400X400
        coordinate_array_wt4(coordinate,:) = (coordinate-1)*d + IJ4; %dX61, stores radius 4 neighbour wts
        coordinate_array(coordinate,:) = IJ; %dX6, to retrieve selected activation from 400X1, eg, MIlinear
        coordinate_array4(coordinate,:) = IJ4; %400X61
                
    end
end

transpose = MI';
MIlinear = transpose(:);        %400X1
PI_transpose = PI';
PIlinear = PI_transpose(:); % 400X1
%--------------------------------------------------------------------------

%THE LEARNING CYCLES FOR THE FIRST INPUT 

for z=1:20
    
    rand('state', sum(100*clock));   %Sets the state of generator to the clock each time
    point =  1+ floor(20*rand(2,1)) ;  % 2X1
    neighbours7([1:3 5:7]) = Calculate_IJarray1(point(1),point(2)); %retrieves the 6 neighbours of point 
    neighbours7(4) = (point(1)-1)*20+point(2);   %1X7, calculate linear coordinate corresponding to point
    input_activation_linear = zeros(400,1);
    input_activation_linear(neighbours7)=0.03;   %400X1, apply activation patch of 0.03
    %temp_input = reshape(input_activation_linear, 20, 20); %reshapes the 1X400 into 20X20 column-wise
    %input_activation = temp_input'; %transpose to get row-wise 20X20
    %MIlinear = MIlinear+input_activation_linear; %apply initial stimulation, at this time prop stimulation is 0.
    
    for a=1:12
        
        %TRANSMISSION OF ACTIVATION FROM PI TO MI, RADIUS 4.
        
        PIMI_weights_transpose = PIMI_weights';
        PIMI_weights_selected = PIMI_weights_transpose(coordinate_array_wt4); % 400X61 
        PIMI_activation_selected = PIlinear(coordinate_array4);
        MIlinear_temp = update_activation_copy(MIlinear, PIMI_weights_selected, PIMI_activation_selected, 1, 0.0001, 0.2, 3, 0.6, input_activation_linear);
        MIlinear = MIlinear_temp;
        %MI = (reshape(MItemp, 20, 20))'; %reshapes 400X1 to 20X20
        
        %---------------------------------------------------------
             
        %TRANSMISSION OF ACTIVATION WITHIN MI
        
        Mweights_transpose = Mweights'; %1X400 row vector
        Mweights_selected = Mweights_transpose(coordinate_array_wt1); %creates a 400X6 matrix storing the weight values
        Mactivation_selected = MIlinear(coordinate_array);
        MItemp = update_activation_copy(MIlinear, Mweights_selected, Mactivation_selected, 1, 0.0001, 0.2, 3, 0.4, input_activation_linear);
        MIlinear=MItemp;
        %MI = (reshape(MItemp, 20, 20))'; %reshape 400X1 into 20X20 and then transpose to row-wise, as reshape is done column-wise
        %-----------------------------------------------------------------------------------------------
        
        %TRANSMISSION OF ACTIVATION FROM MI TO LOWER MOTOR NEURON LAYER
        
        MIlinear_transpose = MIlinear'; %1X400 row vector
        motor_selected_act = repmat(MIlinear_transpose, [6 1]);  %6X400
        motor_temp = update_activation_copy(motor, motor_weights, motor_selected_act, 2, 0.0001, 0.2, 1, 0.05, 0);
        motor = motor_temp;
        %-----------------------------------------------------------------------------------------------
        
        %TRANSMISSION OF ACTIVATION FROM LMN TO PROPRIOCEPTIVE INPUT
        
        for b=(1:2:6)
            Pinput(1,b) = sin((pi/4)*(1-(motor(b)-motor(b+1)))); %stores the length of agonist muscle
            Pinput(2,b) = motor(b) + 0.1*Pinput(1,b); %stores the tension values of the agonist    
            Pinput(1,b+1) = cos((pi/4)*(1-(motor(b)-motor(b+1)))); %length of antagonist muscle
            Pinput(2,b+1) = motor(b+1) + 0.1*Pinput(1,b+1);      %tension of antagonist muscle
        end
        
        %---------------------------------------------------------------------------------------------
        
        %TRANSMISSION OF ACTIVATION FROM PROPRIOCEPTIVE INPUT TO PI
        
        %Proprioceptive layer or PI - has 20X20 elements and is fully connected to
        %Pinput layer
        
        Pinput_tiled = repmat((Pinput(:))', [d 1]); %400X12
        PIlinear_temp = update_activation_copy(PIlinear, PtoPI_weights, Pinput_tiled, 1, 0.1, 0.4, 5, 0.8, 0) %400X1
        PIlinear = PIlinear_temp;
        %PI = (reshape(PItemp, 20, 20))'; %reshapes 400X1 to 20X20
        %---------------------------------------------------------------------------------------------
        
        %TRANSMISSION OF ACTIVATION WITHIN PI
        
        Pweights_transpose = Pweights';
        Pweights_selected = Pweights_transpose(coordinate_array_wt1);   %creates a 400X6 matrix storing the weight values       							
        Pactivation_selected = PIlinear(coordinate_array);  %400X6 storing the activation of neighbours of each neuron
        PIlinear_temp = update_activation_copy(PIlinear, Pweights_selected, Pactivation_selected, 1, 0.0001, 0.4, 5, 0.8, 0);
        PIlinear = PIlinear_temp;
        %PI = (reshape(PItemp, 20, 20))'; %reshapes 400X1 to 20X20
        %----------------------------------------------------------------------------------------------
        
        
    end %120 activation stabilization cycles
    
    motor_weights = learning(motor_weights, motor_selected_act, motor, 0.32, 0.1); %MI to LMN
    PtoPI_weights = learning(PtoPI_weights, Pinput_tiled, PIlinear, 0, 0.2) ;
    PIMI_weights_selected  = learning(PIMI_weights_selected, PIMI_activation_selected, MIlinear, 0, 0.2);
    PIMI_weights_transpose(coordinate_array_wt4) = PIMI_weights_selected; %reset the selected weights in PIMI_weights to new value
        
end %2000 input pattern cycles
               
printMIoutput(motor_weights, 1, 0.32)        
%printPIinput(PI, 'E', 0.4)  UNCOMMENT TO PRINT PI INPUT MAP FOR EXTENSOR LENGTH     
        





