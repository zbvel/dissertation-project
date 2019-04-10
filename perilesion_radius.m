function radius_array = perilesion_radius(radius, corner, dim1, dim2, lesion_coordinates)
%function radius_array = perislesion_radius(radius, corner, dim1, dim2,
%lesion_coordinates)
%radius can be 1 or 2
%corner is a 2d array, 2X1, storing the coordinates of the LH corner of the
%lesion patch (lesion_corner)
%function returns the linear array of coordinates in the radius specified
%dim1 gives the no: of rows in lesion
%dim2 gives num of columns
%lesion_coordinates gives the linear coordinates of the lesions as a matrix

if (radius == 1)
    start_corner = corner - 1; %2X1
    dimension1 = dim1 + 2;
    dimension2 = dim2 + 2;
    lesion_i = zeros(1,dimension1);
    lesion_j = zeros(1,dimension2);
    elements = (dimension1 * dimension2) - (dim1 * dim2);
    radius_array = zeros(1,elements);
end

if (radius==2)
    start_corner = corner -2;
    dimension1 = dim1 + 4;
    dimension2 = dim2 + 4;
    lesion_i = zeros(1,dimension1);
    lesion_j = zeros(1,dimension2);
    elements = (dimension1 * dimension2) - (dim1 * dim2);
    radius_array = zeros(1,elements);
end
  
%coordinates = zeros(dimension1, dimension2);

lesion_i = (start_corner(1):1:start_corner(1)+dimension1-1);
greateri = find(lesion_i>20);
lesion_i(greateri) = lesion_i(greateri)-20;
lesseri = find(lesion_i<1);
lesion_i(lesseri) = lesion_i(lesseri)+20;

lesion_j = (start_corner(2):1:start_corner(2)+dimension2-1);
greaterj = find(lesion_j>20);
lesion_j(greaterj) = lesion_j(greaterj)-20;
lesserj = find(lesion_j<1);
lesion_j(lesserj) = lesion_j(lesserj)+20;

lesion_i_linear = ((lesion_i-1).*20)';% linear vector of row start points i.e. dim1X1
lesion_i_linear_tiled = repmat(lesion_i_linear, [1 dimension2]);
lesion_j_tiled = repmat(lesion_j, [dimension1 1]);
coordinates = lesion_i_linear_tiled + lesion_j_tiled; %linear coord of lesions
radius_array = setdiff(coordinates(:), lesion_coordinates(:)); %remove the lesioned coordinates from the set
radius_array = radius_array'; %transpose to get column vector as act_linear

