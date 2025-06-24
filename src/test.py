import tifffile
vol = tifffile.imread("/home/mathias/PycharmProjects/BoneSimulation/src/utils/output_3d.tiff")
print(vol.shape)
