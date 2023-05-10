using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OptitrackCollect
{
#pragma warning disable IDE1006
    public record Vector3(double x, double y, double z);
    public record Quat(double x, double y, double z, double w);
    public record TrajPoint(Vector3 position, Quat rotation, double timestamp);
    public record OptitrackPoint(string object_id, TrajPoint data);
#pragma warning restore IDE1006
}
