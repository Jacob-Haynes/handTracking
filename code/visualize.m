function visualize(str)
% Demo 1: Visualize a sequence.
%
% Author: Sebastian Nowozin <Sebastian.Nowozin@microsoft.com>


% disp(['Visualizing sequence P3_2_8_p29']);
fprintf('Visualizing sequence %s/n',str)

[X,Y,tagset]=load_file(str);
T=size(X,1);	% Length of sequence in frames

% Animate sequence
h=axes;
for ti=1:T
	skel_vis(X,ti,h);
	drawnow;
	pause(1/30);
	cla;
end
end
