function plot_traj(pos, rot, vel)

facing = zeros(size(pos));
for i = 1:size(facing, 1)
    r = squeeze(rot(i, :, :));
    facing(i, :) = (r * [1; 0; 0]).';
end
quiver3(pos(:,1),pos(:,2),pos(:,3),facing(:,1),facing(:,2),facing(:,3));
hold on
vel = max(min(vel,0.2),-0.2);
quiver3(pos(:,1),pos(:,2),pos(:,3),vel(:,1),vel(:,2),vel(:,3));
axis equal;
axis vis3d;
xlabel('x');
ylabel('y');
zlabel('z');

end

