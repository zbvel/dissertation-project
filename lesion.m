%The stage is implemented by changing the number of cycles within the
%before and after loops below.
%Size is implemented by entering the correct values for dimension1 (dim1)
%and dim2, for eg, 2% size (8 numbers affected) will have dim1=4 and
%dim2=2.
%return the final MI output map
%dim1 is number of affected neurons along one dimension(row)
%dim2 is number of affected neurons along other direction(column)
%dim1*dim2 gives total affected neurons and should be equal to 
%400*(percentage_affected/100)
tic
rand('state', sum(100*clock));
%----------The MI layer - has 20X20 elements - declarations----------------
%%% WITHIN MI

MI = zeros(20,20); %The MI layer of 400 elements
MIlinear_temp = zeros(400,1); %MI in linear form
Mweights_zeros = zeros(400,400);  % create a zeros matrix of same size as intra layer weights matrix
Mweights = 0.4 * (ones(400,400)); %the intra-layer weights matrix
Mweights_selected = zeros(400,6); %to store the neighbour weight values for each element
Mactivation_selected = zeros(400,6); %to store the activations of 6 neighbours 
coordinate_array = zeros(400,6); %to store the retrieved coordinates (i,j) of neighbours
coordinate_array_wt1 = zeros(400,6); %stores the linearised wt-coordinate values from 1-160000

% PI to MI

PIMI_weights_zeros = zeros(400,400);  % create a zeros matrix of same size as weights matrix 400X400
PIMI_weights = 0.1 + (1-0.1) * rand(400,400); %weights from PI to MI
PIMI_weights_selected = zeros(400,61); %to store wt to each neighbour for every element
PIMI_activation_selected = zeros(400,61); %to store the activations of 61 neighbours 
coordinate_array4 = zeros(400,61);   %neighbour coordinates - 61neighbours
coordinate_array_wt4 = zeros(400,61); %to stores wts to 61 neighbours for every element, radius4

%-----------Lower Motor Neuron Layer - declarations-------------------------
%%%
%Lower Motor Neuron Layer or 'motor' - has 6 elements storing the
%activation received by each of the 6 muscles from MI layer

motor = zeros(6,1);
motor_weights = 0.1 + (1-0.1)* rand(6,400);   %stores the inter-layer weights from MI layer to motor

%----------Proprioceptive Input Layer - declarations-----------------------
%%%
% The Proprioceptive Input layer - has 12 elements (2X6) storing the length
% and tension values for each of the 6 muscles

Pinput = zeros(2,6); %row1 stores length and row2 stores tension of each of the 6 muscles in above order
PtoPI_weights = 0.1 + (1-0.1)*rand(400,12);  %stores inter-layer weights from Pinput to PI; 400X12

%----------PI Layer has 20X20 elements - declarations------------------------------
%%%
PI = zeros(20,20); %activation in PI
PIlinear_temp = zeros(400,1); %linear PI
Pweights = 0.4 * (ones(400,400));        %stores intra-layer weights; 400X400
Pweights_zeros = zeros(400,400);  % create a zeros matrix of same size as weights matrix 400X400
Pweights_selected = zeros(400,6); %selected intra-layer weights
Pactivation_selected = zeros(400,6); %to store the activations of 6 neighbours 

%-----------Input patch -----------------------------------------------------------

%Produce the initial activation patch to be applied to MI layer
 
neighbours7 = zeros(1,7);
input_activation_linear = zeros(400,1); %to linearly store the input values, equivalent to transpose-linear of a 20X20 input matrix;
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
old_MIlinear = MIlinear;
PI_transpose = PI';
PIlinear = PI_transpose(:);     % 400X1
old_PIlinear = PIlinear;
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

PIMI_weights_transpose = PIMI_weights';
PIMI_weights_selected = PIMI_weights_transpose(coordinate_array_wt4); % 400X61

Mweights_transpose = Mweights'; %1X400 row vector
Mweights_selected = Mweights_transpose(coordinate_array_wt1); %400X6

Pweights_transpose = Pweights';
Pweights_selected = Pweights_transpose(coordinate_array_wt1); %400X6

%Print MI output map before training
MI_output_before = printMIoutput(prev_motor_weights, 1, 0.897);

%--------------------------------------------------------------------------

%THE LEARNING AND ACTIVATION UPDATE CYCLES 

for z=1:1600 %learning cycles
    
    
    rand('state', sum(100*clock));     %Sets the state of generator to the clock each time
    point =  1+ floor(20*rand(2,1)) ;  % 2X1
    neighbours7([1:3 5:7]) = Calculate_IJarray1(point(1),point(2)); %retrieves the 6 neighbours of point 
    neighbours7(4) = (point(1)-1)*20+point(2);   %1X7, calculate linear coordinate corresponding to point
    input_activation_linear(:)=0;                %has to be initialized each time, to prevent multiple inputs
    input_activation_linear(neighbours7)=0.3;   %400X1, apply activation patch of 0.03, radius 1
      
        
    for a=1:120 %activation stabilization cylces
        
        %TRANSMISSION OF ACTIVATION FROM PI TO MI, RADIUS 4.
        Mactivation_selected = MIlinear(coordinate_array);       %retrieves the updated activation values of neighbours        
        PIMI_activation_selected = PIlinear(coordinate_array4);  %retrieves the updated activation values 
        MIlinear_temp = update_activation(PIMI_weights, MIlinear, PIMI_weights_selected, PIMI_activation_selected, coordinate_array4, 1, 0.0001, -2, 3, 0.6, input_activation_linear);
        MIlinear = MIlinear + MIlinear_temp;
        
        %---------------------------------------------------------
             
        %TRANSMISSION OF ACTIVATION WITHIN MI
       
        Mactivation_selected = MIlinear(coordinate_array); 
        MItemp = update_activation(Mweights, MIlinear, Mweights_selected, Mactivation_selected, coordinate_array, 1, 0.0001, -2, 3, 0.4, input_activation_linear);
        MIlinear =  MIlinear +  MItemp;
        
        %-----------------------------------------------------------------------------------------------
        
        %TRANSMISSION OF ACTIVATION FROM MI TO LOWER MOTOR NEURON LAYER
        
        MIlinear_transpose = MIlinear'; %1X400 row vector
        motor_selected_act = repmat(MIlinear_transpose, [6 1]);  %6X400
        motor_temp = update_activation(motor_weights, motor, motor_weights, motor_selected_act, coordofneighbours1, 2, 0.0001, -2, 1, 0.05, 0);
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
        PIlinear = update_activation(PtoPI_weights, PIlinear, PtoPI_weights, Pinput_tiled, coordofneighbours2, 1, 0.1, -4, 5, 0.8, 0); %400X1
        PIlinear = PIlinear + PIlinear_temp;
        
        %---------------------------------------------------------------------------------------------
        
        %TRANSMISSION OF ACTIVATION WITHIN PI        
        				
        Pactivation_selected = PIlinear(coordinate_array);  %400X6 storing the activation of neighbours of each neuron
        PIlinear = update_activation(Pweights, PIlinear, Pweights_selected, Pactivation_selected, coordinate_array, 1, 0.0001, -4, 5, 0.8, 0);
        PIlinear =   PIlinear + PIlinear_temp;
        
        %----------------------------------------------------------------------------------------------
        
        
    end %120 activation stabilization cycles
    
    fprintf('%s','a')
    if(rem(z,50)== 0), fprintf('\n'), end;
    %tic
    motor_weights = learning(motor_weights, motor_selected_act, motor, 0, 0.1); %MI to LMN
    PtoPI_weights = learning(PtoPI_weights, Pinput_tiled, PIlinear, 0, 0.2) ;
    PIMI_weights_selected  = learning(PIMI_weights_selected, PIMI_activation_selected, MIlinear, 0, 0.2);
    PIMI_weights_transpose(coordinate_array_wt4) = PIMI_weights_selected; %reset the selected weights in PIMI_weights to new value
    
end %2000 input pattern cycles


MIlinear_before_lesion = MIlinear; %save the value just before introducing lesion
motor_weights_beforelesion = motor_weights;
MI_output_beforelesion = printMIoutput(motor_weights, 1, 0.897) ;

rand('state', sum(100*clock));     %Sets the state of generator to the clock each time
lesion_corner = [7;20];
lesion_coord = zeros(dim1, dim2);
lesion_i = zeros(1,dim1);
lesion_j = zeros(1,dim2);
lesion_i = (lesion_corner(1):1:lesion_corner(1)+dim1-1);
greateri = find(lesion_i>20);
lesion_i(greateri) = lesion_i(greateri)-20;
lesseri = find(lesion_i<1);
lesion_i(lesseri) = lesion_i(lesseri)+20;

lesion_j = (lesion_corner(2):1:lesion_corner(2)+dim2-1);
greaterj = find(lesion_j>20);
lesion_j(greaterj) = lesion_j(greaterj)-20;
lesserj = find(lesion_j<1);
lesion_j(lesserj) = lesion_j(lesserj)+20;

lesion_i_linear = ((lesion_i-1).*20)';% linear vector of row start points i.e. dim1X1
lesion_i_linear_tiled = repmat(lesion_i_linear, [1 dim2]);
lesion_j_tiled = repmat(lesion_j, [dim1 1]);
lesion_coord = lesion_i_linear_tiled + lesion_j_tiled; %linear coord of lesions
lesion_coord = lesion_coord'; %transpose to get column vector as act_linear


MIlinear(lesion_coord) = 0; %set those elements to zero activation
(reshape(MIlinear, 20, 20))' %to view lesion patch
motor_weights(:,lesion_coord)=0 %sever the weight connections from the affected neurons to motor
PIMI_weights(lesion_coord,:)=0; %sever the weight connections to the affected neurons from PI
PIMI_weights_transpose = PIMI_weights';
PIMI_weights_selected = PIMI_weights_transpose(coordinate_array_wt4); % 400X61
Mweights(:,lesion_coord) = 0;
Mweights(lesion_coord,:)=0;
   
   

for z = 1:400
    
    rand('state', sum(100*clock));     %Sets the state of generator to the clock each time
    point =  1+ floor(20*rand(2,1)) ;  % 2X1
    neighbours7([1:3 5:7]) = Calculate_IJarray1(point(1),point(2)); %retrieves the 6 neighbours of point 
    neighbours7(4) = (point(1)-1)*20+point(2);   %1X7, calculate linear coordinate corresponding to point
    input_activation_linear(:)=0;                %has to be initialized each time, to prevent multiple inputs
    input_activation_linear(neighbours7)=0.3;   %400X1, apply activation patch of 0.03, radius 1
    %fprintf('%s\n','input')
    %toc %end of point
    
       
    if(z==2) 
        MIlinear_after_lesion = MIlinear; %save the value just before introducing lesion
        motor_weights_afterlesion = motor_weights;
        MI_output_afterlesion = printMIoutput(motor_weights, 1, 0.897) ;
    end
    
    if(z==100)
        MIlinear_100 = MIlinear; %save the value just before introducing lesion
        motor_weights_100 = motor_weights;
        MI_output_100 = printMIoutput(motor_weights, 1, 0.897) ;
    end
    
      
    for a=1:120 %activation stabilization cylces
        MIlinear(lesion_coord) = 0; %set those elements to zero activation
        %TRANSMISSION OF ACTIVATION FROM PI TO MI, RADIUS 4.
        Mactivation_selected = MIlinear(coordinate_array);       %retrieves the updated activation values of neighbours        
        PIMI_activation_selected = PIlinear(coordinate_array4);  %retrieves the updated activation values 
        MIlinear_temp = update_activation_lesion(PIMI_weights, MIlinear, PIMI_weights_selected, PIMI_activation_selected, coordinate_array4, 1, 0.0001, -2, 3, 0.6, input_activation_linear);
        MIlinear = MIlinear + MIlinear_temp;
        MIlinear(lesion_coord) = 0;
        
        %---------------------------------------------------------
             
        %TRANSMISSION OF ACTIVATION WITHIN MI
        
        Mactivation_selected = MIlinear(coordinate_array); 
        MItemp = update_activation_lesion(Mweights, MIlinear, Mweights_selected, Mactivation_selected, coordinate_array, 1, 0.0001, -2, 3, 0.4, input_activation_linear);
        MIlinear =  MIlinear +  MItemp;
        MIlinear(lesion_coord) = 0;
        
        %-----------------------------------------------------------------------------------------------
        
        %TRANSMISSION OF ACTIVATION FROM MI TO LOWER MOTOR NEURON LAYER
        
        MIlinear_transpose = MIlinear'; %1X400 row vector
        motor_selected_act = repmat(MIlinear_transpose, [6 1]);  %6X400
        motor_temp = update_activation_lesion(motor_weights, motor, motor_weights, motor_selected_act, coordofneighbours1, 2, 0.0001, -2, 1, 0.05, 0);
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
        PIlinear = update_activation(PtoPI_weights, PIlinear, PtoPI_weights, Pinput_tiled, coordofneighbours2, 1, 0.1, -4, 5, 0.8, 0); %400X1
        PIlinear = PIlinear + PIlinear_temp;
       
        %---------------------------------------------------------------------------------------------
        
        %TRANSMISSION OF ACTIVATION WITHIN PI        
        					
        Pactivation_selected = PIlinear(coordinate_array);  %400X6 storing the activation of neighbours of each neuron
        PIlinear = update_activation(Pweights, PIlinear, Pweights_selected, Pactivation_selected, coordinate_array, 1, 0.0001, -4, 5, 0.8, 0);
        PIlinear =   PIlinear + PIlinear_temp;
        
        %----------------------------------------------------------------------------------------------
        
        
    end %120 activation stabilization cycles
   
    fprintf('%s','a')
    if(rem(z,50)== 0), fprintf('\n'), end;
    
    motor_weights = learning(motor_weights, motor_selected_act, motor, 0, 0.1); %MI to LMN
    motor_weights(:,lesion_coord)=0; %sever the weight connections from the affected neurons to motor
    
    PtoPI_weights = learning(PtoPI_weights, Pinput_tiled, PIlinear, 0, 0.2) ;
      
    PIMI_weights_selected  = learning(PIMI_weights_selected, PIMI_activation_selected, MIlinear, 0, 0.2);
    PIMI_weights_selected(lesion_coord,:)=0; %sever the weight connections to the affected neurons from PI
    PIMI_weights_transpose(coordinate_array_wt4) = PIMI_weights_selected; %reset the selected weights in PIMI_weights to new value
    
    if(z==800), motor_weights_afterlesion800 = motor_weights, end;
     
end %2000 input pattern cycles
  
(reshape(MIlinear, 20, 20))' %to view lesion patch at the end, verify lesions still exist
%MI output maps for values above 0.897 in motor_weights(appropriate row)
MI_output_after = printMIoutput(motor_weights, 1, 0.897)    
MIlinear_final = MIlinear; %save the value just before introducing lesion
motor_weights_final = motor_weights;

toc








