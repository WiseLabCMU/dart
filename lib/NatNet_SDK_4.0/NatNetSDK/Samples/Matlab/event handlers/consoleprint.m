function consoleprint( ~ , evnt )
	% the input argument, evnt, is a c structure half converted into a
	% matlab structure.
	data = evnt.data;
	rbnum = 1;

	% get the quaternion values of the rigid body and convert to Euler
	% Angles, right handed, global, XYZ order. Should be the same output
	% from Motive.
	q = quaternion( data.RigidBodies( rbnum ).qx, data.RigidBodies( rbnum ).qy, data.RigidBodies( rbnum ).qz, data.RigidBodies( rbnum ).qw );
	qRot = quaternion( 0, 0, 0, 1);
	q = mtimes( q, qRot);
	a = EulerAngles( q , 'zyx' );
	eulerx = a( 1 ) * -180.0 / pi;
	eulery = a( 2 ) * -180.0 / pi;
	eulerz = a( 3 ) * 180.0 / pi;
	
	% print the data to the command window for every 100th frame
    if (rem(evnt.data.iFrame,200) == 0)
        fprintf( 'Frame #%5d (Rigid Body ID:%1d)\n', evnt.data.iFrame ,data.RigidBodies(rbnum).ID);

        fprintf( '\tX:%0.1fmm ', data.RigidBodies( rbnum ).x * 1000 );
        fprintf( '\tY:%0.1fmm ', data.RigidBodies( rbnum ).y * 1000 );
        fprintf( '\tZ:%0.1fmm\n', data.RigidBodies( rbnum ).z * 1000 );

        fprintf( '\tPitch:%0.1f ', eulerx );
        fprintf( '\tYaw:%0.1f ', eulery );
        fprintf( '\tRoll:%0.1f\n', eulerz );
    end
end
