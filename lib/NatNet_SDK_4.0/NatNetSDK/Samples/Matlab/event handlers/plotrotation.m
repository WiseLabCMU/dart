
function plotrotation( ~ , evnt )
	% The eventcallback function executs each time a frame of mocap data is delivered.
	% to Matlab. Matlab will lag if the data rate from the Host is too high.
	%A simple animated line graph displays the x, y, z rotation of the first rigid body in the Host.

	
	% Note - This callback uses the gobal variables qx, qy, qz from the setup.m script.
	% Run setup.m instead.
	global rx
	global ry
	global rz
	global a2
	
	% local variables
	persistent frame2
	persistent lastframe2
	rbnum = 1;	
	
	% Get the frame
	frame2 = double( evnt.data.iFrame );
	if ~isempty( frame2 ) && ~isempty( lastframe2 )
		if frame2 < lastframe2
			rx.clearpoints;
			ry.clearpoints;
			rz.clearpoints;
		end
	end


	% Get the rb rotation
	rb = evnt.data.RigidBodies( rbnum );
	qx = rb.qx;
	qy = rb.qy;
	qz = rb.qz;
	qw = rb.qw;
	
	q = quaternion( qx, qy, qz, qw );
	qRot = quaternion( 0, 0, 0, 1);
	q = mtimes( q, qRot);
	a = EulerAngles( q , 'zyx' );
	eulerx = a( 1 ) * -180.0 / pi;
	eulery = a( 2 ) * 180.0 / pi;
	eulerz = a( 3 ) * -180.0 / pi;
	

	% Fill the animated line's queue with the rb rotation
	frame = frame2;
	rx.addpoints( frame , eulerx );
	ry.addpoints( frame , eulery );
	rz.addpoints( frame , eulerz );
	
	
	% set the figure
	set(gcf,'CurrentAxes', a2)
	
	
	% Dynamically move the axis of the graph
	axis( [ -240 + frame , 20 + frame , -180 , 180 ] );

	
	% Draw the data to a figure
	drawnow
	
	
	% Update lastframe
	lastframe2 = frame2;
	
	
end  % eventcallback1
