from  PyQt5 import QtWidgets as qtw
from PyQt5 import QtGui as qtg
from PyQt5 import QtCore as qtc
import os

import tensorflow as tf

from utils import label_map_util

@dataclass
class App:
    model:tf.Graph=None
    labelmap:dict=None
    image:qtg.QImage=None

app = App()

class MainW(qtw.QWidget):
    def __init__(self):
        super(MainW,self).__init__()
        self.setLayout(qtw.QVBoxLayout())
        self.layout().addWidget(ModelW("Drop model *.pb file here"))
        self.layout().addWidget(LabelMapW())
        #self.layout().addWidget(qtw.QPushButton("Run model"))

class ImageW(qtw.QScrollArea):
    def __init__(self):
        super(ImageW,self).__init__()
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
    def __init__(self,*args):
        super(ModelW,self).__init__(*args)
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
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

class LabelMapW(qtw.QListWidget):
    def __init__(self,*args):
        super(LabelMapW,self).__init__(*args)
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


if __name__ == "__main__":

    app =qtw.QApplication([])
    mainw = MainW()
    mainw.show()
    imagew = ImageW()
    imagew.show()
    app.exec()