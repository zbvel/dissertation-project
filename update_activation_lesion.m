function new_activation = update_activation_lesion(weight_matrix, act_linear, selected_weights, selected_activation, coordinatesofneighbours, p, q, Cs, M, Cp, input)
%update the activation values
%act_linear --- the 'to' activation matrix, transposed and linearized to give column vector
%selected_weights --- stores the weights from each neuron to its neighbours, size: num-of-elementsXnum-of-neighbours
%selected_activation --- stores the activations of the neighbours for each element, size is same as above
%p, q, Cs, M and Cp --- activation parameters specified in the reference paper
%input --- (400X1), the radius-1 activation patch applied to the MI layer, zero for other layers
%weight_matrix ---- is of the size: num-of-elementsXnum-of-elements-in-input, stores
%the weights
%coordinatesofneighbours --- is of the same size as selected_weights

elements = size(selected_weights, 1);    %number of rows in weight matrix gives the number of elements in that layer
neighbours = size(selected_weights, 2);  %num of columns gives num of neighbours
allneighbours = size(weight_matrix,2); %all neighbours including those that it is not connected to, i.e. all elements in input layer
fraction = zeros(elements, neighbours);

%calculate first part of equation
part1 = Cs * act_linear ;                         %Parameter Cs for MI = -2, 400X1
part2 = M - act_linear;                           %Parameter M for MI = 3, 400X1
linear_pq = ((act_linear).^p + q);                %Parameter Cp for MI = 0.4, 400X1 matrix

%tile activation matrix from 400X1 to 400Xneighbours
alltiled_linear_pq = repmat(linear_pq, [1 allneighbours]); %num-of-elements X num-of-elements in input layer
%tiled_linear_pq = repmat(linear_pq, [1 neighbours]), this can also be
%given by----
tiled_linear_pq = alltiled_linear_pq(:,(1:neighbours)); %num-of-elements X num-of-neighbours

denominator = alltiled_linear_pq .* weight_matrix;  %the weight_matrix has wts to non-neighbours set to zero
denominator_sum = sum(denominator, 1); %sum all rows, 1Xnum-of-elements in input layer, 
                                      % this gives the sum of the competition for every element in the input layer
competition = denominator_sum(coordinatesofneighbours); %select the competition values for each connected neighbour,
                                                      %size of coordinatesofneighbours = num-of-elementsXnum-of-neighbours
individual_act = tiled_linear_pq .* selected_weights; %num-of-elementsXnum-of-neighbours

%find zero values in competition to avoid division by zero
affected = find(competition == 0);  %this gives the indices of (element, affected_neighbour), so that the competition from that neighbour is zero
intact = find(competition ~= 0);  %those indices which are intact
intact_individual_act = individual_act(intact); %removes individual activation corresponding to the affected neighbours, this balances the matrix
intact_competition = competition(intact); %remove the zero values from denominator

fraction(intact) = intact_individual_act ./ intact_competition;   
fraction(affected)=0; %set those indices where competition is zero, to zero

numerator = Cp * fraction .* selected_activation; %400X6
init_act = sum(numerator, 2);       %sum all columns in a row to get initial activation for each element: num-of-elementsX1
part3 = init_act + input;           %num-of-elementsX1

%The complete equation for calculating the activation of each element,
%num-of-elementsX1 matrix
new_activation = part1 + (part2 .* (part3)); %num-of-elementsX1
zero_index = find(new_activation<0);
new_activation(zero_index)=0;
end

%-------------------------------------------------------------------------------------------------


	

