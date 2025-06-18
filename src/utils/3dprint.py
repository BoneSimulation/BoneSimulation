import sys
import os
import pyvista as pv
import meshio

def convert_mesh_to_stl(input_path, output_path):
    if not os.path.isfile(input_path):
        print(f"Fehler: Datei nicht gefunden: {input_path}")
        return False

    print(f"Lade Mesh von: {input_path}")
    mesh = pv.read(input_path)

    if isinstance(mesh, pv.UnstructuredGrid):
        print("Extrahiere Oberfläche aus UnstructuredGrid...")
        mesh = mesh.extract_surface()

    print("Bereinige Mesh...")
    mesh = mesh.clean()

    faces = mesh.faces.reshape((-1, 4))[:, 1:4]

    meshio_mesh = meshio.Mesh(points=mesh.points, cells=[("triangle", faces)])

    print(f"Speichere als STL: {output_path}")
    meshio.write(output_path, meshio_mesh, file_format="stl")

    print("Fertig.")
    return True


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python mesh_to_stl_converter.py <input_mesh_file> <output_stl_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    success = convert_mesh_to_stl(input_file, output_file)
    if not success:
        sys.exit(1)
