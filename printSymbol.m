function printSymbol(matrix, muscle_number, symbol1, symbol2)
rows = size(matrix, 1);
columns = size(matrix, 2);

for x=1:rows
    for y=1:columns
        if matrix(x,y) == muscle_number
            fprintf('%s', symbol1)
            fprintf(' ')
        else
            fprintf('%s', symbol2)
            fprintf(' ')
        end
    end
    fprintf('\n')
    
end
fprintf('\n')
            