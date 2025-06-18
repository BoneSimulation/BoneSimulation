def export_to_calculix(mesh_points, mesh_elements, material_props, output_path, fix_z_plane=True, load_z_plane=True,
                       load_value=100.0):
    """
    Exportiert eine CalculiX .inp Datei.
    - mesh_points: (N x 3)
    - mesh_elements: (M x 4) Tetraeder
    - material_props: {'E': ..., 'nu': ...}
    - fix_z_plane: fixiert alle Knoten auf z==min_z
    - load_z_plane: belastet alle Knoten auf z==max_z
    - load_value: Kraft pro Knoten in Z
    """
    import numpy as np

    z_coords = mesh_points[:, 2]
    min_z = np.min(z_coords)
    max_z = np.max(z_coords)

    tolerance = 1e-5
    fixed_nodes = [i + 1 for i, z in enumerate(z_coords) if abs(z - min_z) < tolerance]
    loaded_nodes = [i + 1 for i, z in enumerate(z_coords) if abs(z - max_z) < tolerance]

    with open(output_path, "w") as f:
        f.write("*HEADING\n")
        f.write("Knochen-Simulation\n")

        f.write("*NODE\n")
        for i, (x, y, z) in enumerate(mesh_points, start=1):
            f.write(f"{i}, {x}, {y}, {z}\n")

        f.write("*ELEMENT, TYPE=C3D4\n")
        for i, elem in enumerate(mesh_elements, start=1):
            elem_str = ", ".join(str(n) for n in elem)
            f.write(f"{i}, {elem_str}\n")

        f.write("*MATERIAL, NAME=Bone\n")
        f.write("*ELASTIC\n")
        f.write(f"{material_props['E']}, {material_props['nu']}\n")

        f.write("*SOLID SECTION, ELSET=AllElements, MATERIAL=Bone\n")
        f.write("AllElements\n")

        f.write("*ELSET, ELSET=AllElements, GENERATE\n")
        f.write(f"1, {len(mesh_elements)}, 1\n")

        # --- AUTOMATISCHE Randbedingungen ---
        if fix_z_plane and fixed_nodes:
            f.write("*BOUNDARY\n")
            for n in fixed_nodes:
                f.write(f"{n}, 1, 3, 0.0\n")

        # --- AUTOMATISCHE Lasten ---
        if load_z_plane and loaded_nodes:
            f.write("*CLOAD\n")
            force_per_node = load_value / len(loaded_nodes)
            for n in loaded_nodes:
                f.write(f"{n}, 3, {force_per_node}\n")

        # --- Analyse Step ---
        f.write("*STEP\n")
        f.write("*STATIC\n")
        f.write("*END STEP\n")

    print(f"CalculiX Input gespeichert: {output_path}")
    print(f"Fixierte Knoten (z=min): {len(fixed_nodes)}")
    print(f"Belastete Knoten (z=max): {len(loaded_nodes)}")
