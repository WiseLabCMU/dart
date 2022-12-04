from scipy.io import loadmat
import torch


class Map3d:
    def __init__(self, mapfile):
        map = loadmat(mapfile)
        self.x = torch.Tensor(map['x']).double()
        self.y = torch.Tensor(map['y']).double()
        self.z = torch.Tensor(map['z']).double()
        self.v = torch.Tensor(map['v']).double()
        resx, resy, resz = self.v.shape
        ppmx = resx / (self.x[-1, -1, -1] - self.x[0, 0, 0])
        ppmy = resy / (self.y[-1, -1, -1] - self.y[0, 0, 0])
        ppmz = resz / (self.z[-1, -1, -1] - self.z[0, 0, 0])
        self.res = torch.Tensor([resx, resy, resz])
        self.ppm = torch.Tensor([ppmx, ppmy, ppmz])

    def get_sigma(self):
        return self.sample_nearest.__get__(self, self.__class__)

    def sample_nearest(self, xyz):
        uvw = torch.round(xyz * self.ppm + self.res / 2).long()
        mask = torch.logical_and(0 <= uvw, uvw < self.res).all(dim=1)
        output = torch.zeros(mask.shape).double()
        output[mask] = self.v[uvw[mask, 0], uvw[mask, 1], uvw[mask, 2]]
        return output

    def sample_trilinear(self, xyz):
        batch, _ = xyz.shape
        return torch.zeros((batch,))


class Trajectory:
    def __init__(self, trajfile):
        traj = loadmat(trajfile)
        self.timestamp = torch.Tensor(traj['timestamp']).double()
        self.position = torch.Tensor(traj['position']).double()
        self.orientation = torch.Tensor(traj['orientation']).double()
        self.velocity = torch.Tensor(traj['velocity']).double()
        self.acceleration = torch.Tensor(traj['acceleration']).double()
        self.angularVelocity = torch.Tensor(traj['angularVelocity']).double()


if __name__ == '__main__':
    map = Map3d('data/map.mat')
    sigma = map.get_sigma()
    xyz = torch.Tensor([[-10,-10,-10],[0.1,0.1,0.1],[0.75,0.75,0.75]])
    batch = sigma(xyz)
    print(batch)

    traj = Trajectory('data/traj.mat')
    print(traj.orientation.shape)
