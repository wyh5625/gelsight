import matplotlib.pyplot as plt

import pyvista as pv
from pyvista import examples

# Load an interesting example of geometry
mesh = examples.load_random_hills()
mesh.plot(screenshot='hills.png')

# Establish geometry within a pv.Plotter()
p = pv.Plotter()
p.add_mesh(mesh, color=True)
p.show()

zval = p.get_image_depth()
zval_filled_by_42s = p.get_image_depth(fill_value=42.0)

plt.figure()
plt.imshow(zval)
plt.colorbar(label='Distance to Camera')
plt.title('Depth image')
plt.xlabel('X Pixel')
plt.ylabel('Y Pixel')
plt.show()