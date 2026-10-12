"""Synthetic part with known features for the guided-selection tests (epic guided-*, 2026-10-12).

Plate 60x40x10 with a 10 mm through hole, one vertical edge filleted r=4 and a countersunk hole
cskHole(6, 12, 90). Tessellated face by face, so every triangle carries the B-rep face it came from:
that is the ground truth a user (or the oracle) would select.
"""
import numpy as np
import trimesh

KIND = {"GeomAbs_Plane": "plane", "GeomAbs_Cylinder": "cylinder", "GeomAbs_Cone": "cone",
        "GeomAbs_Sphere": "sphere", "GeomAbs_Torus": "torus"}


def part():
    import cadquery as cq

    return (cq.Workplane("XY").box(60, 40, 10)
            .edges("|Z and >X and >Y").fillet(4)
            .faces(">Z").workplane().hole(10)
            .faces(">Z").workplane().center(-18, 0).cskHole(6, 12, 90)
            .val())


def tessellate(shape, lin=0.02, ang=0.2):
    """-> (trimesh, face_of_triangle[int], kinds[str per B-rep face], radius[float|None per face])."""
    from OCP.BRep import BRep_Tool
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS

    s = shape.wrapped if hasattr(shape, "wrapped") else shape
    BRepMesh_IncrementalMesh(s, lin, False, ang, True)
    tris, owner, kinds, radii = [], [], [], []
    e = TopExp_Explorer(s, TopAbs_FACE)
    fi = 0
    while e.More():
        f = TopoDS.Face_s(e.Current())
        ad = BRepAdaptor_Surface(f)
        k = KIND.get(str(ad.GetType()).split(".")[-1], "other")
        kinds.append(k)
        radii.append(ad.Cylinder().Radius() if k == "cylinder" else ad.Sphere().Radius() if k == "sphere"
                     else ad.Torus().MinorRadius() if k == "torus" else None)
        loc = TopLoc_Location()
        t = BRep_Tool.Triangulation_s(f, loc)
        tr = loc.Transformation()
        P = np.array([[*(lambda p: (p.X(), p.Y(), p.Z()))(t.Node(i).Transformed(tr))]
                      for i in range(1, t.NbNodes() + 1)])
        rev = f.Orientation() == TopAbs_REVERSED
        for i in range(1, t.NbTriangles() + 1):
            a, b, c = t.Triangle(i).Get()
            tris.append(P[[a - 1, c - 1, b - 1]] if rev else P[[a - 1, b - 1, c - 1]])
            owner.append(fi)
        fi += 1
        e.Next()
    tri = np.array(tris)
    mesh = trimesh.Trimesh(tri.reshape(-1, 3), np.arange(len(tri) * 3).reshape(-1, 3), process=False)
    return mesh, np.array(owner), kinds, radii


def volume(shape):
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps

    g = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape.wrapped if hasattr(shape, "wrapped") else shape, g)
    return g.Mass()


def read_step(path):
    """-> (shape, [(kind, radius-or-None)], n_solids, valid)."""
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.STEPControl import STEPControl_Reader
    from OCP.TopAbs import TopAbs_FACE, TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    r = STEPControl_Reader()
    r.ReadFile(str(path))
    r.TransferRoots()
    s = r.OneShape()
    faces = []
    e = TopExp_Explorer(s, TopAbs_FACE)
    while e.More():
        ad = BRepAdaptor_Surface(TopoDS.Face_s(e.Current()))
        k = KIND.get(str(ad.GetType()).split(".")[-1], "other")
        faces.append((k, ad.Cylinder().Radius() if k == "cylinder" else ad.Sphere().Radius() if k == "sphere"
                      else ad.Torus().MinorRadius() if k == "torus" else None))
        e.Next()
    n = 0
    e = TopExp_Explorer(s, TopAbs_SOLID)
    while e.More():
        n += 1
        e.Next()
    return s, faces, n, BRepCheck_Analyzer(s).IsValid()


if __name__ == "__main__":
    m, owner, kinds, radii = tessellate(part())
    print(len(m.faces), "triangles,", len(kinds), "faces:", list(zip(kinds, radii, strict=True)))
    print("watertight after merge:", m.copy().merge_vertices() or m.is_watertight, round(volume(part()), 2))
