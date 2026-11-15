import sys
import numpy as np
from PyQt5 import QtWidgets, QtGui
import pyqtgraph.opengl as gl

fmt = QtGui.QSurfaceFormat()
fmt.setRenderableType(QtGui.QSurfaceFormat.OpenGL)
fmt.setVersion(2, 1)
fmt.setProfile(QtGui.QSurfaceFormat.NoProfile)
QtGui.QSurfaceFormat.setDefaultFormat(fmt)

app = QtWidgets.QApplication(sys.argv)

view = gl.GLViewWidget()
view.setWindowTitle("OpenGL diagnostic")
view.setCameraPosition(distance=10)

grid = gl.GLGridItem()
view.addItem(grid)

line = gl.GLLinePlotItem(
    pos=np.array([[0, 0, 0], [2, 2, 2]], dtype=np.float32),
    color=(1, 0, 0, 1),
    width=3
)
view.addItem(line)

view.show()
sys.exit(app.exec_())
