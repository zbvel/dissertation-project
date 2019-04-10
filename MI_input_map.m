%Prints the input maps formed in the MI layer before and after training
%Any one of the Proprioceptive input elements is set to a non-zero value 
%at a time, and the corresponding elements in MI with activation above a
%threshold are measured
%Maps are printed before and after training

%----------The MI layer - has 20X20 elements - declarations----------------
%%% WITHIN PI
MI = 0.03 + (3-0.03) * rand(20,20);  % activation values in MI layer 0<a<M
MIlinear_temp = zeros(400,1);
Mweights_zeros = zeros(400,400);  % create a zeros matrix of same size as intra layer weights matrix
Mweights = 0.1 + (1-0.1)*rand(400,400); %the intra-layer weights matrix
Mweights_selected = zeros(400,6); %to store the neighbour weight values for each element
Mactivation_selected = zeros(400,6); %to store the activations of 6 neighbours 
coordinate_array = zeros(400,6); %to store the retrieved coordinates (i,j) of neighbours
coordinate_array_wt1 = zeros(400,6); %stores the linearised wt-coordinate values from 1-160000

% PI to MI

PIMI_weights_zeros = zeros(400,400);  % create a zeros matrix of same size as weights matrix 400X400
PIMI_weights = 0.1 + (1-0.1)*rand(400,400); %the PI to MI inter-layer weights matrix
PIMI_weights_selected = zeros(400,61); %to store wt to each neighbour for every element
PIMI_activation_selected = zeros(400,61); %to store the activations of 61 neighbours 
coordinate_array4 = zeros(400,61);   %neighbour coordinates - 61neighbours
coordinate_array_wt4 = zeros(400,61); %to stores wts to 61 neighbours for every element, radius4

%-----------Lower Motor Neuron Layer - declarations-------------------------
%%%
%Lower Motor Neuron Layer or 'motor' - has 6 elements storing the
%activation received by each of the 6 muscles from MI layer

motor = 0.03 + (1-0.03) * rand(6,1) ;   %stores the activation values of 6 muscles in this order; E,F,AB,AD,O,C
motor_weights = 0.1 + (1-0.1)*rand(6,400);   %stores the inter-layer weights from MI layer to motor

%----------Proprioceptive Input Layer - declarations-----------------------
%%%
% The Proprioceptive Input layer - has 12 elements (2X6) storing the length
% and tension values for each of the 6 muscles

Pinput = zeros(2,6); %row1 stores length and row2 stores tension of each of the 6 muscles in above order
%Pinput = Pinput * 0.03;
PtoPI_weights = 0.1 + (1-0.1)*rand(400,12);  %stores inter-layer weights from Pinput to PI; 400X12

%----------PI Layer has 20X20 elements - declarations------------------------------
%%%
PI = 0.03 + (5-0.03)*rand(20,20);              %activation values stored here; 20X20
PIlinear_temp = zeros(400,1);
Pweights = 0.1 + (1-0.1)*rand(400,400);        %stores intra-layer weights; 400X400
Pweights_zeros = zeros(400,400);  % create a zeros matrix of same size as weights matrix 400X400
Pweights_selected = zeros(400,6); %selected intra-layer weights
Pactivation_selected = zeros(400,6); %to store the activations of 6 neighbours 

%-----------Input to PI from arm  -----------------------------------------------------------

%Produce the initial activation - a single element of Pinput at a time
 
input_to_arm = zeros(2,6); %set all elements to zero
input_to_arm(1,1) = 0.3; %set the activation of Extensor length to 0.3
                    
%----------------------------------------------------------------------------------

for x=1:20
    for y=1:20
        IJ = Calculate_IJarray1(x,y);  %1X6, coordinates of 6 neighbours
        IJ4 = Calculate_IJarray4(x,y); %1X61, coordinates of 61 neighbours
        coordinate = (x-1)*20 + y;     %linear value of (x,y) from 1:400
        coordinate_array(coordinate,:) = IJ; %400X6, coordinates of neighbours for every element
        coordinate_array4(coordinate,:) = IJ4; %400X61, same as above, but for 61 neighbours
        coordinate_array_wt1(coordinate,:) = (coordinate-1)*400 + IJ; %400X6, to retrieve wts of connected neighbours from 400X400, hence value between 1-1600000
        coordinate_array_wt4(coordinate,:) = (coordinate-1)*400 + IJ4; %400X61, same as above, but for 61 neighbours
        Mweights_zeros(coordinate, IJ)= 1; % set those elements corresponding to coordinates in IJ to 1 (from 0)  
        Pweights_zeros(coordinate,IJ)=1;   %same as above
        PIMI_weights_zeros(coordinate, IJ4)=1; 
        
    end
end

%sort all row elements in all the coordinate arrays
coordinate_array_wt1 = sort(coordinate_array_wt1,2);
coordinate_array_wt4 = sort(coordinate_array_wt4,2);
coordinate_array = sort(coordinate_array,2);
coordinate_array4 = sort(coordinate_array4,2);

index1 = find(Mweights_zeros<1); %find positions with value 0, corresponds to positions of non-connected neighbours
index2 = find(PIMI_weights_zeros<1);
index3 = find(Pweights_zeros<1);

Mweights(index1)=0; %set weights to non-neighbours to zero
PIMI_weights(index2)=0;
Pweights(index3)=0;

%Following is to determine linearized activation values of MI and PI
%layers(running left to right along rows)
transpose = MI';
MIlinear = transpose(:);        %400X1
PI_transpose = PI';
PIlinear = PI_transpose(:);     % 400X1
%--------------------------------------------------------------------------
%the coordinate array for MI to LMN----
%as its fully connected, the neighbours will always be the same, 
%running from 1to 400 for each of the 6 motor neurons
n1 = (1:400); %each neuron in LMN has 400 neighbours its connected to in MI, 
%i.e. each element in LMN receives activation from 400 elements in MI
coordofneighbours1 = repmat(n1, [6 1]); %there are 6 elements in LMN,num-of-elementsXnum-of-neighbours, 6X400
%--------------------------------------------------------------------------
%The same way for connection from Pinput to PI
n2 = (1:12) ; %each neuron in PI receives input from all 12 neurons in Pinput
coordofneighbours2 = repmat(n2, [400 1]);  %there are 400 neurons in PI, num-of-elementsXnum-of-neighbours, 400X12
%--------------------------------------------------------------------------

%Print the PI input map once before the training begins
index = find(MIlinear>0.4);
MIlinear_print_before = zeros(400,1);
MIlinear_print_before(index)=1; %set those elements to 1 whose activation values are above threshold
MIlinear_print_before = (reshape(MIlinear_print_before, 20, 20))'; %reshapes 400X1 to 20X20

%--------------------------------------------------------------------------

%THE LEARNING AND ACTIVATION UPDATE CYCLES 

for z=1:2000 %learning cycles
    
    %Apply input to Pinput layer
    Pinput = Pinput + input_to_arm;
    
    %Extract the weights to neighbours based on the coordinate array from
    %before, i.e running from 1 to 160000
    PIMI_weights_transpose = PIMI_weights';    
    PIMI_weights_selected = PIMI_weights_transpose(coordinate_array_wt4); % 400X61 
    
    Mweights_transpose = Mweights'; %1X400 row vector        
    Mweights_selected = Mweights_transpose(coordinate_array_wt1); %400X6
    
    Pweights_transpose = Pweights';        
    Pweights_selected = Pweights_transpose(coordinate_array_wt1); %400X6   
    
    
    
    for a=1:120 %activation stabilization cylces
        
        %TRANSMISSION OF ACTIVATION FROM PI TO MI, RADIUS 4.
        
        Mactivation_selected = MIlinear(coordinate_array);       %retrieves the updated activation values of neighbours        
        PIMI_activation_selected = PIlinear(coordinate_array4);  %retrieves the updated activation values 
        MIlinear_temp = update_activation(PIMI_weights, MIlinear, PIMI_weights_selected, PIMI_activation_selected, coordinate_array4, 1, 0.0001, -0.75, 3, 0.6, 0);
        MIlinear =  MIlinear + MIlinear_temp;
        %MI = (reshape(MIlinear, 20, 20))'; %reshapes 400X1 to 20X20, 
        %then transpose to row-wise, as reshape is done column-wise
        %can be used to assign linear values back to 20X20 MI matrix
        
        %---------------------------------------------------------
             
        %TRANSMISSION OF ACTIVATION WITHIN MI
        
        Mactivation_selected = MIlinear(coordinate_array);          
        MItemp = update_activation(Mweights, MIlinear, Mweights_selected, Mactivation_selected, coordinate_array, 1, 0.0001, -0.75, 3, 0.4, 0);
        MIlinear= MIlinear + MItemp;
        %MI = (reshape(MItemp, 20, 20))'; %reshape 400X1 into 20X20 
        %-----------------------------------------------------------------------------------------------
        
        %TRANSMISSION OF ACTIVATION FROM MI TO LOWER MOTOR NEURON LAYER
        
        MIlinear_transpose = MIlinear'; %1X400 row vector
        motor_selected_act = repmat(MIlinear_transpose, [6 1]);  %6X400
        motor_temp = update_activation(motor_weights, motor, motor_weights, motor_selected_act, coordofneighbours1, 2, 0.0001, -0.75, 1, 0.05, 0);
        motor =  motor + motor_temp;
        %-----------------------------------------------------------------------------------------------
        
        %TRANSMISSION OF ACTIVATION FROM LMN TO PROPRIOCEPTIVE INPUT
        
        for b=(1:2:6)
            Pinput(1,b) =  sin((pi/4)*(1-(motor(b)-motor(b+1)))); %stores the length of agonist muscle
            Pinput(2,b) =  motor(b) + 0.1*Pinput(1,b); %stores the tension values of the agonist    
            Pinput(1,b+1) =  cos((pi/4)*(1-(motor(b)-motor(b+1)))); %length of antagonist muscle
            Pinput(2,b+1) =  motor(b+1) + 0.1*Pinput(1,b+1);      %tension of antagonist muscle
        end
        
              
        %---------------------------------------------------------------------------------------------
        
        %TRANSMISSION OF ACTIVATION FROM PROPRIOCEPTIVE INPUT TO PI
        
        Pinput_tiled = repmat((Pinput(:))', [400 1]); %400X12
        PIlinear_temp = update_activation(PtoPI_weights, PIlinear, PtoPI_weights, Pinput_tiled, coordofneighbours2, 1, 0.1, -0.75, 5, 0.8, 0) %400X1
        PIlinear = PIlinear + PIlinear_temp;
        %PI = (reshape(PItemp, 20, 20))'; %reshapes 400X1 to 20X20
        %---------------------------------------------------------------------------------------------
        
        %TRANSMISSION OF ACTIVATION WITHIN PI        
        					
        Pactivation_selected = PIlinear(coordinate_array);  %400X6 storing the activation of neighbours of each neuron
        PIlinear_temp = update_activation(Pweights, PIlinear, Pweights_selected, Pactivation_selected, coordinate_array, 1, 0.0001, -0.75, 5, 0.8, 0);
        PIlinear =  PIlinear + PIlinear_temp;
        %PI = (reshape(PItemp, 20, 20))'; %reshapes 400X1 to 20X20
        %----------------------------------------------------------------------------------------------
        
        
    end %120 activation stabilization cycles
    
    motor_weights = learning(motor_weights, motor_selected_act, motor, 0.32, 0.1); %MI to LMN
    PtoPI_weights = learning(PtoPI_weights, Pinput_tiled, PIlinear, 0, 0.2) ;
    PIMI_weights_selected  = learning(PIMI_weights_selected, PIMI_activation_selected, MIlinear, 0, 0.2);
    PIMI_weights_transpose(coordinate_array_wt4) = PIMI_weights_selected; %reset the selected weights in PIMI_weights to new value
       
end %2000 input pattern cycles

%Print the PI input map after training is completed
index = find(MIlinear>0.4);
MIlinear_print_after = zeros(400,1);
MIlinear_print_after(index)=1; %set those elements to 1 whose activation values are above threshold
MIlinear_print_after = (reshape(MIlinear_print_after, 20, 20))'; %reshapes 400X1 to 20X20



