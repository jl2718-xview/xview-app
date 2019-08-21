from  PyQt5 import QtWidgets as qtw
from PyQt5 import QtGui as qtg
from PyQt5 import QtCore as qtc
import os,sys

import tensorflow as tf
print("PYTHONPATH=",os.environ["PYTHONPATH"])
from object_detection.utils import label_map_util




class MainW(qtw.QWidget):
    def __init__(self,app):
        super(MainW,self).__init__()
        self.app = app
        self.setLayout(qtw.QVBoxLayout())
        self.layout().addWidget(ModelW(app,"Drop model *.pb file here"))
        self.layout().addWidget(LabelMapW(app))
        runbutton = qtw.QPushButton("Run model",self)
        runbutton.clicked.connect(self.runModel)
        self.layout().addWidget(runbutton)
    def runModel(self):
        if not self.app.graph: return



class ImageW(qtw.QScrollArea):
    def __init__(self,app):
        super(ImageW,self).__init__()
        self.app = app
        self.image = qtg.QImage()
        self.setWidget(qtw.QLabel())
        self.setWidgetResizable(True)
        #self.widget().setBaseSize(300,300)
        self.setAcceptDrops(True)
    def dragEnterEvent(self, e: qtg.QDragEnterEvent) -> None:
        if os.path.exists(e.mimeData().urls()[0].toLocalFile()):
            e.accept()
        else:
            e.ignore()
    def dropEvent(self, e: qtg.QDropEvent) -> None:
        self.loadImage(e.mimeData().urls()[0].toLocalFile())
    def loadImage(self,path:str)->None:
        image = qtg.QImage(path)
        if image.isNull():
            return
        self.image=image
        pixmap = qtg.QPixmap()
        pixmap.convertFromImage(self.image)
        self.widget().setPixmap(pixmap)
        self.widget().setAlignment(qtc.Qt.AlignHCenter | qtc.Qt.AlignVCenter)
    def runModel(self):
        graph = tf.get_default_graph()



class ModelW(qtw.QLabel):
    def __init__(self,app,*args):
        super(ModelW,self).__init__(*args)
        self.app = app
        self.setAcceptDrops(True)
        self.setBaseSize(100,200)
    def dragEnterEvent(self, e: qtg.QDragEnterEvent) -> None:
        if os.path.exists(e.mimeData().urls()[0].toLocalFile()):
            e.accept()
        else:
            e.ignore()
    def dropEvent(self, e: qtg.QDropEvent) -> None:
        self.loadModel(e.mimeData().urls()[0].toLocalFile())
    def loadModel(self,path:str):
        self.setText(path)
        self.app.graph = tf.Graph()
        with self.app.graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

class LabelMapW(qtw.QListWidget):
    def __init__(self,app,*args):
        super(LabelMapW,self).__init__(*args)
        self.app = app
        self.setAcceptDrops(True)
        self.setDragEnabled(True)
        self.setSelectionMode(qtw.QAbstractItemView.ExtendedSelection)
        #self.setDragDropMode()
        self.addItem("Drop LabelMap here")
    def dragEnterEvent(self, e: qtg.QDragEnterEvent) -> None:
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
        else:
            e.ignore()
    def dragMoveEvent(self, e: qtg.QDragMoveEvent) -> None:
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
        else:
            e.ignore()
    def dropEvent(self, e: qtg.QDropEvent) -> None:
        self.loadMap(e.mimeData().urls()[0].toLocalFile())
    def loadMap(self,path):
        self.clear()
        for l in label_map_util.load_labelmap(path).item:
            self.addItem(str(l.id) +": "+l.display_name)


class MyApp(qtw.QApplication):
    def __init__(self,*args):
        super(MyApp,self).__init__(*args)
        # windows
        self.mainw = MainW(self)
        self.mainw.show()
        self.imagew = ImageW(self)
        self.imagew.show()
        # globals
        self.graph = None
        self.labelmap = None
        self.image = None

if __name__ == "__main__":

    app = MyApp([])
    app.exec()