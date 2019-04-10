function IJarray = Calculate_IJarray1(i, j)
%Calculates the linear coordinates of neighbours of a given element
%the given element is specified by its x and y coordinates
%the neighbour coordinates correspond to the 6 neighbours,
%the element itself is not considered

Jarray=zeros(1,7);  %to store the j coordinates of neighbours
Iarray=zeros(1,7);  %to stores the i coordinates of neighbours
IJarray_temp = zeros(1,7);
IJarray = zeros(1,6); %to store the linear coordinates w.r.t (i,j)

if (rem(i,2)==0), J=[0, 1, -1, 0, 1, 0, 1];  %if element belongs to an even row
else J=[-1, 0, -1, 0, 1, -1, 0];, end
I=[-1, -1, 0, 0, 0, 1, 1];

Jarray=J+j;  %to get corresponding j values of neighbours
Iarray=I+i;  %gives i coordinates of neighbours

%in the following steps, the torus is implemented
%this is done by finding coordintes greater than 20 or less than 1,
%20 is added to values less than 1, eg, 0 becomes 20, -1 becomes 19, etc
indexless=find(Jarray<1); 
Jarray(indexless)=Jarray(indexless)+20;
indexlessI=find(Iarray<1);
Iarray(indexlessI)=Iarray(indexlessI)+20;

%20 is subtracted from the coordinates greater than 20, eg, 21 becomes 1,
%22 becomes 2, and so on.
indexmore = find(Jarray>20);
Jarray(indexmore)=Jarray(indexmore)-20;
indexmoreI = find(Iarray>20);
Iarray(indexmoreI)=Iarray(indexmoreI)-20;

%transform the two dimensions of i and j running from 1 to 20 each, into a one
%dimenion array running from 1 to 400
IJarray_temp = (Iarray-1)*20 + Jarray;
IJarray = IJarray_temp([1:3 5:end]); %remove element itself from list, now its 1X6


