function MB = storage(x)
% return size of doubles matrix of size x in megabytes
%this is something i found in a couple implementations for 
%faster computation with variable tree sizes
MB= prod(x)*8/1024/1024;

end
