function IJarray = Calculate_IJarray4(i,j)
%Calculates the neighbour coordinates of an element within a radius of 4
%the element is specified using its i and j coordinate values
%the coordinates of neighbours are linearized  running from 1 to 400

Jarray=zeros(1, 61);
Iarray=zeros(1,61);
IJarray=zeros(1,61);
prevcoord = 0;

if (rem(i,2)==0), initial=[-2, -2, -3, -3, -4]; %if element belongs to even row
else initial=[-2, -3, -3, -4, -4];, end
I=[-4, -3, -2, -1, 0];


for a=1:5      %radius of 4 +1 = 5
    Jarray(prevcoord+1) = j+initial(a);
    Iarray(prevcoord+1) = i+I(a);
    Iarray(61-prevcoord) = i-I(a);
    
    tempJ = Jarray(prevcoord+1);
    tempI = Iarray(prevcoord+1);
    tempIR = Iarray(61-prevcoord);
    prevcoord = prevcoord+1;
    for b=1:(4+a-1)
        Jarray(prevcoord+1) = tempJ+1;
        tempJ = Jarray(prevcoord+1);
        Iarray(prevcoord+1) = tempI;
        tempI = Iarray(prevcoord+1);
        Iarray(61-prevcoord) = tempIR;
        tempIR = Iarray(61-prevcoord);
        prevcoord=prevcoord+1;
    end
end

%j values being replicated 
Jarray(36:43)=Jarray(19:26);
Jarray(44:50)=Jarray(12:18);
Jarray(51:56)=Jarray(6:11);
Jarray(57:61)=Jarray(1:5);

%following implements the torus
%coordinates greater than 20 and less than 1 are transformed,
%20 is subtracted from values greater than 20, eg, 21 becomes 1, etc
%20 is added to coordinates less than 1, eg, -1 becomes 19, and so on
indexless=find(Jarray<1);
Jarray(indexless)=Jarray(indexless)+20;
indexlessI=find(Iarray<1);
Iarray(indexlessI)=Iarray(indexlessI)+20;

indexmore = find(Jarray>20);
Jarray(indexmore)=Jarray(indexmore)-20;
indexmoreI = find(Iarray>20);
Iarray(indexmoreI)=Iarray(indexmoreI)-20;

%transform the two dimensions of i and j running from 1 to 20 each, into a one
%dimenion array running from 1 to 400
IJarray = (Iarray-1)*20 + Jarray;




