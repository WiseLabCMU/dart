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
        return self.sample_trilinear.__get__(self, self.__class__)

    def sample_nearest(self, xyz):
        return self.uvw2val(self.xyz2uvw_nearest(xyz))

    def sample_trilinear(self, xyz):
        uvw = self.xyz2uvw_floor(xyz)
        t = (xyz - self.uvw2xyz(uvw)) * self.ppm
        r = 1 - t
        output = self.uvw2val(uvw + torch.Tensor([0, 0, 0]).long()) * r[:, 0] * r[:, 1] * r[:, 2]
        output += self.uvw2val(uvw + torch.Tensor([0, 0, 1]).long()) * r[:, 0] * r[:, 1] * t[:, 2]
        output += self.uvw2val(uvw + torch.Tensor([0, 1, 0]).long()) * r[:, 0] * t[:, 1] * r[:, 2]
        output += self.uvw2val(uvw + torch.Tensor([0, 1, 1]).long()) * r[:, 0] * t[:, 1] * t[:, 2]
        output += self.uvw2val(uvw + torch.Tensor([1, 0, 0]).long()) * t[:, 0] * r[:, 1] * r[:, 2]
        output += self.uvw2val(uvw + torch.Tensor([1, 0, 1]).long()) * t[:, 0] * r[:, 1] * t[:, 2]
        output += self.uvw2val(uvw + torch.Tensor([1, 1, 0]).long()) * t[:, 0] * t[:, 1] * r[:, 2]
        output += self.uvw2val(uvw + torch.Tensor([1, 1, 1]).long()) * t[:, 0] * t[:, 1] * t[:, 2]
        return output

    def xyz2uvw_nearest(self, xyz):
        return torch.round(xyz * self.ppm + self.res / 2).long()

    def xyz2uvw_floor(self, xyz):
        return torch.floor(xyz * self.ppm + self.res / 2).long()

    def uvw2xyz(self, uvw):
        return (uvw - self.res / 2) / self.ppm

    def uvw2val(self, uvw):
        mask = torch.logical_and(0 <= uvw, uvw < self.res).all(dim=1)
        output = torch.zeros(mask.shape, dtype=torch.double)
        output[mask] = self.v[uvw[mask, 0], uvw[mask, 1], uvw[mask, 2]]
        return output


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
    xyz = torch.Tensor([[-10,-10,-10],[0.1,0.1,0.1],[0.75,0.75,0.75],[-.54, 0, 0]])
    batch = sigma(xyz)
    print(batch)

    traj = Trajectory('data/traj.mat')
    print(traj.orientation.shape)
