function new_activation = update_activation_copy(act_linear, selected_weights, selected_activation, p, q, Cs, M, Cp, input)
%update the activation values
%act_linear --- the 'to' activation matrix, transposed and linearized to give column vector
%selected_weights --- stores the weights from each neuron to its neighbours, size: num-of-elementsXnum-of-neighbours
%selected_activation --- stores the activations of the neighbours for each element, size is same as above
%p, q, Cs, M and Cp --- activation parameters specified in the reference paper
%input --- (400X1), the radius-1 activation patch applied to the MI layer, zero for other layers


s=20;
d=400;

elements = size(selected_weights, 1);    %number of rows in weight matrix gives the number of elements in that layer
neighbours = size(selected_weights, 2);  %num of columns gives num of neighbours

%calculate first part of equation
part1 = Cs * act_linear ;                         %Parameter Cs for MI = -2, 400X1
part2 = M - act_linear;                           %Parameter M for MI = 3, 400X1
linear_pq = ((act_linear).^p + q);            %Parameter Cp for MI = 0.4, 400X1 matrix

%tile activation matrix from 400X1 to 400X6 or 400X61
tiled_linear_pq = repmat(linear_pq, [1 neighbours]);

act_weights = tiled_linear_pq .* selected_weights; %400X6
sum_act_weights = sum(act_weights, 1); %sum all rows along each column, 400X6 to 1X6 or 1X61
denominator = repmat(sum_act_weights, [elements 1]); %tile to 400X6 from 1X6, as 1X6 is for one element


%tile the above to be number_of_elementsXnumber_of_neighbours, 400X6
%tiled_linearCp = tiled_linear_pq .* Cp;

num1 = tiled_linear_pq .* selected_weights; %400X6
num2 = num1 ./ denominator;

numerator = Cp * num2 .* selected_activation; %400X6
init_act = sum(numerator, 2);       %sum all columns in a row to get initial activation for each element: num-of-elementsX1
part3 = init_act + input;                   %num-of-elementsX1

%The complete equation for calculating the activation of each element,
%400X1 matrix
new_activation = part1 + (part2 .* (part3)) %num-of-elementsX1
	
%-------------------------------------------------------------------------------------------------


	

