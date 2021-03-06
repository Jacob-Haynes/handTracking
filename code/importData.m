function importData(fileToRead1, varName)
%IMPORTFILE(FILETOREAD1)
%  Imports data from the specified file
%  FILETOREAD1:  file to read

%  Auto-generated by MATLAB on 08-Dec-2014 11:07:35

% Import the file
newData1 = load('-mat', fileToRead1);

% Create new variables in the base workspace from those fields.
vars = fieldnames(newData1);
training = cell(12,1);
% data files start with i=1 at gesture 10, i=3 is gesture 12, i=4 is
% gesture 1
for i = 4:(length(vars)+3)
   % assignin('base', vars{i}, newData1.(vars{i}));
   training{mod(i-3,12)+1} =  newData1.(vars{mod(i,12)+1});
end

assignin('base', varName, training);
