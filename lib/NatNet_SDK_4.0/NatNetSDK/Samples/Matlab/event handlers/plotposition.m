
function plotposition( ~ , evnt )
	% The event callback function executs each time a frame of mocap data is delivered.
	% to Matlab. Matlab will lag if the data rate from the Host is too high.
	% A simple animated line graphs the x, y, z position of the first rigid body on the Host.

	
	% Note - This callback uses the gobal variables px, py, pz from the NatNetEventHandlerSample.
	global px
	global py
	global pz
	global a1
	
	% local variables
	persistent frame1
	persistent lastframe1
	scope = 1.5;
	rbnum = 1;
	
	
	% Get the frame number
	frame1 = double( evnt.data.iFrame );
	if ~isempty( frame1 ) && ~isempty( lastframe1 )
		if frame1 < lastframe1
			px.clearpoints;
			py.clearpoints;
			pz.clearpoints;
		end
	end

	
	% Get the rigid body position
	x = double( evnt.data.RigidBodies( rbnum ).x );
	y = double( evnt.data.RigidBodies( rbnum ).y );
	z = double( evnt.data.RigidBodies( rbnum ).z );
	
	
	% Fill the animated line's queue with the rb position
	frame = frame1;
	px.addpoints( frame , x );
	py.addpoints( frame , y );
	pz.addpoints( frame , z );

	
	% set the figure and subplot to graph the data
	set( gcf , 'CurrentAxes' , a1 )

	
	% Dynamically move the axis of the graph
	axis( [ -240 + frame ,  20 + frame , min( x , min( y , z ) ) - scope , max( x , max( y , z ) ) + scope ] );

	
	% Draw the data to a figure
	drawnow
	
	
	% Update lastframe
	lastframe1 = frame1;
end  % eventcallback1
