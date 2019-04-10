function probability =  neighbour_probability(map, threshold)
%function probability =  neighbour_probability(map, threshold)
%this calculates the probability that the neighbours of a neuron code for a
%particular muscle's length or tension, given that it itself codes for it.
%map is the input or output map being considered
%threshold is as considered in the map

rows = size(map,1);
columns = size(map,2);
transpose = map';
map_linear = transpose(:);
probability_map = zeros(20,20);

for j=1:columns
    for i=1:rows
        if(map(i,j)>= threshold)
            ij_array = Calculate_IJarray1(i,j);
            neighbours = map_linear(ij_array);
            above_num = size(find(neighbours >= threshold),1); %select num of rows because MI_output_after is column vector
            probability_map(i,j) = above_num/6; %total number of neighbours above threshold/num of neighbours
        end
    end
end
probability = sum(sum(probability_map))/400;


         
            
            