function plotrigidbodyvelocity( ~ , evnt )
	% The eventcallback function executs each time a frame of mocap data is delivered.
	% to Matlab. Matlab will lag if the data rate from the Host is too high.
	% A simple animated line graph displays the x, y, z velocity of the first rigid body in the Host.
	
	% Note - This callback uses the gobal variables xv, yv, zv from the setup.m script.
	% Run setup.m instead.
	global xv
	global yv
	global zv
	global a3
	global rbxv
	global rbyv
	global rbzv
	
	persistent frame3
	persistent lastframe3
	persistent time
	persistent lasttime
	persistent lastrbx
	persistent lastrby
	persistent lastrbz

	scope = 10;
	rbnum = 1;
    
	% Get the frame
	frame3 = double( evnt.data.iFrame );
    if ~isempty( frame3 ) || ~isempty( lastframe3 )
            if frame3 < lastframe3
                xv.clearpoints;
                yv.clearpoints;
                zv.clearpoints;
            end
    end
    
	% Get the time of the frame
	time = double( evnt.data.fTimestamp );
	% Get the rb velocity
	rbx = double( evnt.data.RigidBodies( rbnum ).x ); % x position of first rb
	rby = double( evnt.data.RigidBodies( rbnum ).y ); % y position of first rb
	rbz = double( evnt.data.RigidBodies( rbnum ).z ); % z position of first rb
    
	% Compute velocity values
	if ~isempty( lastrbx ) || ~isempty( lastrby ) || ~isempty( lastrbz ) || ~isempty(lasttime)
        
		% Get the time delta
		dt = time - lasttime;

		% Get the velocity
		rbxv = ( rbx - lastrbx ) / dt;
		rbyv = ( rby - lastrby ) / dt;
		rbzv = ( rbz - lastrbz ) / dt;

		% Queue the data
		frame = frame3;
        xv.addpoints( frame , rbxv );
        yv.addpoints( frame , rbyv );
        zv.addpoints( frame , rbzv );


		% set the figure
		set( gcf , 'CurrentAxes' , a3 )

		% Dynamically move the axis of the graph
		axis( [ -240+frame , 20+frame , min(rbxv,min(rbyv,rbxv))-scope , max( rbxv , max( rbyv , rbzv ) )+scope ] );

		% Draw the data to a figure
		drawnow

	end

	% Update lastframe
    lastframe3 = frame3;
	lasttime = time;
	lastrbx = rbx;
	lastrby = rby;
	lastrbz = rbz;
	
end  % eventcallback1