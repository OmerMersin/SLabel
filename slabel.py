from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QListWidget, QHBoxLayout, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QPinchGesture, QGraphicsRectItem, QGraphicsItem, QGraphicsLineItem, QGraphicsTextItem, QInputDialog, QDialog, QLabel, QMessageBox, QCheckBox
from PyQt6.QtGui import QPixmap, QPen, QBrush, QColor, QMovie
from PyQt6.QtCore import Qt, QEvent, QRectF, QPointF, QSizeF, QThread, pyqtSignal, pyqtSlot
import os
import json
import math
import cv2
import yaml
import requests


class MovableDialog(QDialog):
    def mousePressEvent(self, event):
        self.oldPos = event.globalPosition()

    def mouseMoveEvent(self, event):
        delta = event.globalPosition() - self.oldPos
        self.move(int(self.x() + delta.x()), int(self.y() + delta.y()))
        self.oldPos = event.globalPosition()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.close()

class HoverableGraphicsRectItem(QGraphicsRectItem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setAcceptHoverEvents(True)  # Enable hover events

    def hoverEnterEvent(self, event):
        self.setPen(QPen(Qt.GlobalColor.yellow))  # Change the color to yellow when the mouse enters
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.setPen(QPen(Qt.GlobalColor.red))  # Change the color back to red when the mouse leaves
        super().hoverLeaveEvent(event)


class DownloadThread(QThread):
    download_finished = pyqtSignal(object)

    def run(self):
        from ultralytics import YOLO
        model_path = 'yolov8x.pt'

        # Check if the model file exists locally
        if not os.path.exists(model_path):
            # Download the model file
            url = 'https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8x.pt'
            response = requests.get(url)
            with open(model_path, 'wb') as f:
                f.write(response.content)

        # Now you can load the model
        self.model = YOLO(model_path)
        self.download_finished.emit(self.model)

class DetectionThread(QThread):
    detection_finished = pyqtSignal(object)

    def __init__(self, model, img, parent=None):
        super(DetectionThread, self).__init__(parent)
        self.model = model
        self.img = img

    def run(self):
        save_path = os.path.join("temp.jpg")
        self.img.save(save_path)
        image_path = os.path.join("temp.jpg")
        frame = cv2.imread(image_path)
        results = self.model(frame)[0]
        detections = []
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            if score > 0.5:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
                detections.append((x1, y1, x2, y2))
        try:
            os.remove(image_path)
        except:
            pass
        self.detection_finished.emit(detections)


class ImageViewer(QWidget):
    class GraphicsView(QGraphicsView):
        def __init__(self, scene, parent):
            super().__init__(scene)
            self.setMouseTracking(True)  # Enable mouse tracking
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.current_rect_item = None
            self.startPoint = None
            self.ghost_lines = [None, None]
            self.last_mouse_pos = None  # Add this line
            self.meta_key_pressed = False  # Add this line
            self.parent = parent  # Store the parent instance
            self.image_position = QPointF(0, 0)  # Store the position of the image
            self.chosen_rectangles = []  # Add this line to initialize the list of chosen rectangles
            self.model_downloaded = False
            self.mouse_toggle = None

        def scrollContentsBy(self, dx, dy):
            if self.dragMode() == QGraphicsView.DragMode.ScrollHandDrag:
                # When in ScrollHandDrag mode, scroll the view by the given dx and dy
                self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - dx)
                self.verticalScrollBar().setValue(self.verticalScrollBar().value() - dy)
            else:
                # In other modes, use the default behavior
                super().scrollContentsBy(dx, dy)

        def hoverMoveEvent(self, event):
            self.last_mouse_pos = self.mapToScene(event.pos())
            self.draw_ghost_lines(self.mapToScene(event.pos()))

        def mousePressEvent(self, event):
            self.mouse_toggle = event.pos()
            # Get the pixmap item
            pixmap_item = None
            for item in reversed(self.scene().items()):
                if isinstance(item, QGraphicsPixmapItem):
                    pixmap_item = item
                    break

            if pixmap_item is not None:
                pixmap = pixmap_item.pixmap()
                click_point = self.mapToScene(event.pos()) - self.image_position

                # Check if the click occurred within the bounds of the image
                if 0 <= click_point.x() < pixmap.width() and 0 <= click_point.y() < pixmap.height():
                    self.last_mouse_pos = self.mapToScene(event.pos())
                    self.mouse_toggle = self.last_mouse_pos
                    if self.meta_key_pressed:
                        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
                        for item in self.scene().items():
                            # Only set the item as movable if it's not a QGraphicsRectItem
                            if not isinstance(item, QGraphicsRectItem):
                                item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
                        self.parent.is_new_picture = False
                    else:
                        self.startPoint = self.mapToScene(event.pos())
                        self.current_rect_item = HoverableGraphicsRectItem(QRectF(self.startPoint, self.startPoint))
                        self.current_rect_item.setPen(QPen(Qt.GlobalColor.red))
                        self.scene().addItem(self.current_rect_item)

            super().mousePressEvent(event)

        def mouseMoveEvent(self, event):
            self.last_mouse_pos = self.mapToScene(event.pos())
            if not self.meta_key_pressed:
                self.setDragMode(QGraphicsView.DragMode.NoDrag)
                for item in self.scene().items():
                    # Only set the item as not movable if it's not a QGraphicsRectItem
                    if not isinstance(item, QGraphicsRectItem):
                        item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
                if self.startPoint is not None:
                    adjusted_start_point = self.startPoint - self.image_position
                    adjusted_current_point = self.mapToScene(event.pos()) - self.image_position

                    # Get the pixmap item
                    pixmap_item = None
                    for item in reversed(self.scene().items()):
                        if isinstance(item, QGraphicsPixmapItem):
                            pixmap_item = item
                            break

                    if pixmap_item is not None:
                        pixmap = pixmap_item.pixmap()

                        # Adjust the current point to be within the image's dimensions
                        adjusted_current_point.setX(max(min(adjusted_current_point.x(), pixmap.width()), 0))
                        adjusted_current_point.setY(max(min(adjusted_current_point.y(), pixmap.height()), 0))

                        rect = QRectF(adjusted_start_point, adjusted_current_point).normalized()
                        if rect.width() > 5 and rect.height() > 5:  # Only draw the rectangle if its size is bigger than 5 pixels
                            self.current_rect_item.setRect(rect)
                            self.current_rect_item.setBrush(QBrush(QColor(255, 0, 0, 127)))  # Fill the rectangle with a semi-transparent red color
                            # Set the parent of the rectangle to be the pixmap item
                            self.current_rect_item.setParentItem(pixmap_item)
            if self.dragMode() != QGraphicsView.DragMode.ScrollHandDrag and not self.meta_key_pressed:
                self.draw_ghost_lines(self.mapToScene(event.pos()))
            super().mouseMoveEvent(event)


        def draw_rectangle(self, x1, y1, x2, y2):
            rect_item = HoverableGraphicsRectItem(QRectF(QPointF(x1, y1), QPointF(x2, y2)))
            rect_item.setPen(QPen(Qt.GlobalColor.red))
            # Get the QGraphicsPixmapItem that represents the image
            pixmap_item = self.scene().items()[-1]
            # Set the parent of the rectangle to be the pixmap item
            rect_item.setParentItem(pixmap_item)

        def mouseReleaseEvent(self, event):
            if self.meta_key_pressed:
                self.current_mouse_pos = self.mapToScene(event.pos())
                delta = self.current_mouse_pos - self.mouse_toggle
                self.image_position += delta    
            self.startPoint = None
            super().mouseReleaseEvent(event)

        def keyReleaseEvent(self, event):
            if event.key() == Qt.Key.Key_Control:
                self.meta_key_pressed = False
                # Stop drag mode when CMD key is released
                self.setDragMode(QGraphicsView.DragMode.NoDrag)
                for item in self.scene().items():
                    item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
                # Draw ghost lines when CMD key is released
                self.draw_ghost_lines(self.last_mouse_pos)

            super().keyReleaseEvent(event)
            
        def detect_objects(self):
            if self.parent.pixmap.isNull():
                self.dialog.close()
                return
            self.detection_thread = DetectionThread(self.model, self.parent.pixmap)
            self.detection_thread.detection_finished.connect(self.on_detection_finished)
            self.detection_thread.start()

        def on_detection_finished(self, detections):
            self.dialog.close()
            for x1, y1, x2, y2 in detections:
                self.draw_rectangle(x1, y1, x2, y2)

        def label_selected_rectangle(self):
            if self.chosen_rectangles:
                # Prompt the user for the label/class name
                label, ok = QInputDialog.getText(self, 'Label Rectangle', 'Enter class name:')
                if ok and label:
                    pixmap_item = self.scene().items()[-1]
                    for rect in self.chosen_rectangles:
                        # Check if the rect object is still valid
                        if rect.scene() is not None:
                            # Create a QGraphicsTextItem for the label
                            text_item = QGraphicsTextItem(label)
                            # Set the position of the text item to the top left corner of the rectangle
                            text_item.setPos(rect.rect().topLeft())
                            # Set the parent of the text item to be the pixmap item
                            text_item.setParentItem(pixmap_item)
                            # Add the text item to the scene
                            self.scene().addItem(text_item)
                            # Fill the rectangle with green color
                            green_color = QColor("green")
                            green_color.setAlpha(40)  # semi-transparent
                            rect.setBrush(QBrush(green_color))
                    # Clear the list of chosen rectangles
                    class_name = text_item.toPlainText()
                    self.parent.classes_widget.addItem(class_name)
                    self.chosen_rectangles.clear()

        @pyqtSlot(object)
        def on_download_finished(self, model):
            self.dialog.close()
            self.model = model
            self.model_downloaded = True

            # Stop the movie and hide the QLabel after detection
            self.movie.stop()
            self.loading_label.hide()

            self.detect_objects()

        def show_loading_dialog(self):
            # Create a QDialog object
            self.dialog = QDialog(self)
            self.dialog.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
            self.dialog.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)
            self.dialog.setWindowFlag(Qt.WindowType.WindowTransparentForInput, True)
            self.dialog.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

            # Create a QLabel object
            self.loading_label = QLabel(self.dialog)

            # Create a QMovie object
            self.movie = QMovie("/Users/omermersin/Developer/Syhme-Tools/load.gif")

            # Set the movie to the QLabel
            self.loading_label.setMovie(self.movie)

            # Start the movie
            self.movie.start()
            layout = QVBoxLayout()
            layout.addWidget(self.loading_label)
            self.dialog.setLayout(layout)
            self.dialog.show()

        def keyPressEvent(self, event):
            if event.key() == Qt.Key.Key_E:
                self.label_selected_rectangle()
            if event.key() == Qt.Key.Key_W:
                if not self.model_downloaded:
                    self.show_loading_dialog()

                    # Start the download thread
                    self.download_thread = DownloadThread()
                    self.download_thread.download_finished.connect(self.on_download_finished)
                    self.download_thread.start()
                else:
                    self.show_loading_dialog()
                    self.detect_objects()

            if event.key() == Qt.Key.Key_Control:
                self.meta_key_pressed = True
                self.remove_ghost_lines()
            if event.key() == Qt.Key.Key_Q:
                items = self.items(self.mapFromScene(self.last_mouse_pos))
                closest_rect = None
                min_distance = math.inf
                for item in items:
                    if isinstance(item, QGraphicsRectItem):
                        rect = item.rect()
                        distance = min(abs(rect.left() - self.last_mouse_pos.x()), abs(rect.right() - self.last_mouse_pos.x()),
                                       abs(rect.top() - self.last_mouse_pos.y()), abs(rect.bottom() - self.last_mouse_pos.y()))
                        if distance < min_distance:
                            min_distance = distance
                            closest_rect = item
                if closest_rect is not None:
                    self.scene().removeItem(closest_rect)
            if event.key() == Qt.Key.Key_C:
                items = self.items(self.mapFromScene(self.last_mouse_pos))
                closest_rect = None
                min_distance = math.inf
                for item in items:
                    if isinstance(item, QGraphicsRectItem):
                        rect = item.rect()
                        distance = min(abs(rect.left() - self.last_mouse_pos.x()), abs(rect.right() - self.last_mouse_pos.x()),
                                    abs(rect.top() - self.last_mouse_pos.y()), abs(rect.bottom() - self.last_mouse_pos.y()))
                        if distance < min_distance:
                            min_distance = distance
                            closest_rect = item
                if closest_rect is not None:
                    red_color = QColor("red")
                    red_color.setAlpha(40)  # semi-transparent
                    closest_rect.setBrush(QBrush(red_color))
                    self.chosen_rectangles.append(closest_rect)  # Add the chosen rectangle to the list
            if event.key() == Qt.Key.Key_V:
                items = self.items(self.mapFromScene(self.last_mouse_pos))
                for item in items:
                    if isinstance(item, QGraphicsRectItem) and item in self.chosen_rectangles:
                        transparent_color = QColor(0, 0, 0, 0)  # R, G, B, Alpha
                        item.setBrush(QBrush(transparent_color))  # remove the color
                        self.chosen_rectangles.remove(item)  # remove the item from the list
            if event.key() == Qt.Key.Key_Shift:
                for item in self.scene().items():
                    if isinstance(item, QGraphicsRectItem):
                        rect = item.rect()
                        class_name = ''  # Define class_name here
                        if rect.width() > 5 and rect.height() > 5:  # Only include the annotation if its size is bigger than 5 pixels
                            # Create a small rectangle at the top left of the item rectangle
                            label_area = QRectF(rect.topLeft(), QSizeF(1, 1))
                            # Iterate over the items that intersect with this small rectangle
                            for sub_item in self.scene().items(label_area):
                                if isinstance(sub_item, QGraphicsTextItem):
                                    # If the item is a text item, get its text
                                    class_name = sub_item.toPlainText()
                                    break
                    if isinstance(item, QGraphicsRectItem) and item not in self.chosen_rectangles and class_name == '':
                        red_color = QColor("red")
                        red_color.setAlpha(40)  # semi-transparent
                        item.setBrush(QBrush(red_color))
                        self.chosen_rectangles.append(item)  # Add the rectangle to the list

            super().keyPressEvent(event)


        def remove_ghost_lines(self):
            if self.ghost_lines[0] is not None and self.ghost_lines[0] in self.scene().items():
                self.scene().removeItem(self.ghost_lines[0])
            if self.ghost_lines[1] is not None and self.ghost_lines[1] in self.scene().items():
                self.scene().removeItem(self.ghost_lines[1])
            self.ghost_lines = [None, None]
            
        def draw_ghost_lines(self, pos):
            if self.parent.is_new_picture:
                self.image_position = QPointF(0, 0)
            if self.ghost_lines[0] is not None and self.ghost_lines[0] in self.scene().items():
                self.scene().removeItem(self.ghost_lines[0])
            if self.ghost_lines[1] is not None and self.ghost_lines[1] in self.scene().items():
                self.scene().removeItem(self.ghost_lines[1])
            scene_items = self.scene().items()
            if scene_items:  # Check if the list is not empty
                pixmap_item = scene_items[-1]  # Assuming the pixmap item is the last item added to the scene
                if isinstance(pixmap_item, QGraphicsPixmapItem):
                    self.scene().items()[-1].setPos(self.image_position)        
                    pixmap_rect = self.scene().items()[-1].sceneBoundingRect()
                    if pixmap_rect.contains(pos):
                        self.ghost_lines[0] = QGraphicsLineItem(pixmap_rect.left(), pos.y(), pixmap_rect.right(), pos.y())
                        self.ghost_lines[1] = QGraphicsLineItem(pos.x(), pixmap_rect.top(), pos.x(), pixmap_rect.bottom())
                        self.ghost_lines[0].setPen(QPen(Qt.GlobalColor.green, 0, Qt.PenStyle.DashLine))
                        self.ghost_lines[1].setPen(QPen(Qt.GlobalColor.green, 0, Qt.PenStyle.DashLine))
                        self.scene().addItem(self.ghost_lines[0])
                        self.scene().addItem(self.ghost_lines[1])
                    else:
                        # Adjust the position to the nearest edge of the pixmap if it is outside
                        adjusted_pos = pos
                        if pos.x() < pixmap_rect.left():
                            adjusted_pos.setX(pixmap_rect.left())
                        elif pos.x() > pixmap_rect.right():
                            adjusted_pos.setX(pixmap_rect.right())
                        if pos.y() < pixmap_rect.top():
                            adjusted_pos.setY(pixmap_rect.top())
                        elif pos.y() > pixmap_rect.bottom():
                            adjusted_pos.setY(pixmap_rect.bottom())
                        self.ghost_lines[0] = QGraphicsLineItem(pixmap_rect.left(), adjusted_pos.y(), pixmap_rect.right(), adjusted_pos.y())
                        self.ghost_lines[1] = QGraphicsLineItem(adjusted_pos.x(), pixmap_rect.top(), adjusted_pos.x(), pixmap_rect.bottom())
                        self.ghost_lines[0].setPen(QPen(Qt.GlobalColor.green, 0, Qt.PenStyle.DashLine))
                        self.ghost_lines[1].setPen(QPen(Qt.GlobalColor.green, 0, Qt.PenStyle.DashLine))
                        self.scene().addItem(self.ghost_lines[0])
                        self.scene().addItem(self.ghost_lines[1])

    def __init__(self):
        super().__init__()
        self.pixmap = QPixmap()
        self.is_new_picture = True
        

        self.layout = QHBoxLayout()
        self.v_layout = QVBoxLayout()
        self.right_layout = QVBoxLayout()
        self.list_widget = QListWidget()
        self.list_widget.setFixedWidth(200)
        self.classes_widget = QListWidget()
        self.classes_widget.setFixedWidth(200)
        self.image_list = []
        self.current_image_index = 0

        self.open_button = QPushButton('Open directory')
        self.open_button.clicked.connect(self.open_directory)

        self.prev_button = QPushButton('Previous')
        self.prev_button.clicked.connect(self.show_prev_image)

        self.next_button = QPushButton('Next')
        self.next_button.clicked.connect(self.show_next_image)

        self.zoom_in_button = QPushButton('Zoom In')
        self.zoom_in_button.clicked.connect(self.zoom_in)

        self.zoom_out_button = QPushButton('Zoom Out')
        self.zoom_out_button.clicked.connect(self.zoom_out)

        self.export_button = QPushButton('Export Annotations')
        self.export_button.clicked.connect(self.export_annotations)

        self.v_layout.addWidget(self.zoom_in_button)
        self.v_layout.addWidget(self.zoom_out_button)

        self.v_layout.addWidget(self.open_button)
        self.v_layout.addWidget(self.prev_button)
        self.v_layout.addWidget(self.next_button)
        self.v_layout.addWidget(self.export_button)

        self.scene = QGraphicsScene()
        self.view = self.GraphicsView(self.scene, self)
        self.view.viewport().grabGesture(Qt.GestureType.PinchGesture)
        
        self.shortcut_info_button = QPushButton('Shortcut Info')
        self.shortcut_info_button.clicked.connect(self.show_shortcut_info)
        self.v_layout.addWidget(self.shortcut_info_button)

        self.layout.addLayout(self.v_layout)
        self.layout.addWidget(self.view)

        self.right_layout.addWidget(self.list_widget)
        self.right_layout.addWidget(self.classes_widget)
        self.layout.addLayout(self.right_layout)

        self.setLayout(self.layout)
        self.showMaximized()

        self.list_widget.itemClicked.connect(self.list_item_clicked)
        self.classes_widget.itemClicked.connect(self.classes_item_clicked)

        self.export_yolo_button = QPushButton('Export Yolo')
        self.export_yolo_button.clicked.connect(self.export_yolo)
        self.v_layout.addWidget(self.export_yolo_button)
        self.export_yolo_button = QPushButton('Create config.yaml')
        self.export_yolo_button.clicked.connect(self.create_config_yaml)
        self.v_layout.addWidget(self.export_yolo_button)
        self.save_location = None

        self.annotations = []
        self.existing_annotations = []
        self.dir_path = None

        self.save_mode = False
        self.save_mode_checkbox = QCheckBox('Activate Save Mode')
        self.save_mode_checkbox.stateChanged.connect(self.toggle_save_mode)
        self.v_layout.addWidget(self.save_mode_checkbox)


    def open_directory(self):
        dir_path = QFileDialog.getExistingDirectory(self, 'Select Directory')
        if dir_path:
            self.dir_path = dir_path
            # Clear the image list
            self.image_list = []
            # Clear the list widget
            self.list_widget.clear()
            self.image_list = [f for f in os.listdir(dir_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
            if not self.image_list:  # If there are no images in the directory
                return
            self.list_widget.addItems(self.image_list)
            self.show_image(self.dir_path, self.image_list[0])
            self.draw_annotations()  # Draw annotations after opening a directory

    def toggle_save_mode(self):
        self.save_mode = self.save_mode_checkbox.isChecked()

    def show_shortcut_info(self):
        dialog = MovableDialog(self)
        dialog.setWindowTitle("Shortcut Info")
        dialog.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        dialog.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)
        layout = QVBoxLayout()
        layout.addWidget(QLabel(
            "ESC: Shortcut Info\n"
            "Q: Delete selected areas\n"
            "W: Run detection\n"
            "E: Enter class name for selected areas\n"
            "A: Previous Image\n"
            "S: Save Annotation\n"
            "D: Next Image\n"
            "Z: Zoom in\n"
            "X: Zoom out\n"
            "C: Select areas\n"
            "V: Unselect areas\n"
            "CTRL+Drag: Move Image\n"
            "Shift: Select all areas"
        ))
        dialog.setLayout(layout)
        dialog.show()

    def export_yolo(self):
        self.show_loading_dialog()
        if self.dir_path is None:
            QMessageBox.warning(self, "Warning", "First open a directory")
            self.dialog.close()
            return
        # Select a directory to save YOLO annotations
        save_dir = QFileDialog.getExistingDirectory(self, 'Select a directory')
        if not save_dir: #save directory
            self.dialog.close()
            return
        
        # Check if an image folder is currently opened
        if not self.dir_path: # currently open directory
            self.dialog.close()
            return

        # Create a 'labels' directory within the image folder
        labels_dir = os.path.join(save_dir, 'labels')
        os.makedirs(labels_dir, exist_ok=True)

        # Initialize a dictionary to map class names to indexes
        class_to_index = {}

        # Iterate through each annotation
        for annotation in self.existing_annotations:
            if annotation["filename"] in self.image_list:

                class_name = annotation['class']

                # Assign an index to the class if it's not already in the dictionary
                if class_name not in class_to_index:
                    class_to_index[class_name] = len(class_to_index)

                # Calculate YOLO format coordinates
                x_center = (annotation['x'] + annotation['width'] / 2) / self.pixmap.width()
                y_center = (annotation['y'] + annotation['height'] / 2) / self.pixmap.height()
                width = annotation['width'] / self.pixmap.width()
                height = annotation['height'] / self.pixmap.height()

                # Extract filename without extension
                filename, _ = os.path.splitext(annotation['filename'])
                
                # Write YOLO format annotation to TXT file
                with open(os.path.join(labels_dir, f"{filename}.txt"), 'a') as f:
                    f.write(f"{class_to_index[class_name]} {x_center} {y_center} {width} {height}\n")

        # Write class names and their indexes to 'classes.txt'
        classes_file_path = os.path.join(labels_dir, 'classes.txt')
        with open(classes_file_path, 'w') as f:
            for class_name, index in class_to_index.items():
                f.write(f"{index}: {class_name}\n")
        
        # Set the save location for future reference
        self.save_location = classes_file_path
    
        self.dialog.close()

    def create_config_yaml(self):
        self.show_loading_dialog()
        if self.save_location is None:
            QMessageBox.warning(self, "Warning", "First export as YOLO")
            self.dialog.close()
            return
        # Read classes from classes.txt
        classes = {}
        with open(self.save_location, 'r') as f:
            for line in f:
                index, class_name = line.strip().split(': ')
                classes[index] = class_name

        data = {
            'path': "/path/to/dataset",
            'train': "/path/to/dataset/train/images",
            'val': "/path/to/dataset/validation/images",
            'names': classes
        }

        # Let the user choose a directory to save the config.yaml file
        dir_path = QFileDialog.getExistingDirectory(self, 'Select a directory')
        if not dir_path:
            self.dialog.close()
            return

        with open(os.path.join(dir_path, 'config.yaml'), 'w') as outfile:
            yaml.dump(data, outfile, default_flow_style=False, sort_keys=False)

        self.dialog.close()

    def draw_annotations(self):
        self.classes_widget.clear()
        # Load existing annotations
        try:
            with open('annotations.json', 'r') as f:
                self.existing_annotations = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.existing_annotations = []

        # Filter annotations for the current image
        current_image_annotations = [a for a in self.existing_annotations if a['filename'] == self.image_list[self.current_image_index]]

        # Draw rectangles for each annotation
        for annotation in current_image_annotations:
            rect = QRectF(annotation['x'], annotation['y'], annotation['width'], annotation['height'])
            rectangle = self.scene.addRect(rect)
            black_color = QColor("white")
            black_color.setAlpha(60)  # semi-transparent
            rectangle.setBrush(QBrush(QColor(black_color)))  # Fill the rectangle with black color
            if annotation.get('class'):
                class_name = annotation['class']
                if not any(class_name == self.classes_widget.item(i).text() for i in range(self.classes_widget.count())):
                    self.classes_widget.addItem(class_name)
                label = QGraphicsTextItem(class_name, rectangle)
                label.setPos(rectangle.rect().topLeft())

            # Set the parent of the rectangle to be the pixmap item
            pixmap_item = self.scene.items()[-1]
            rectangle.setParentItem(pixmap_item)

    def show_prev_image(self):
        if self.save_mode:
            self.export_annotations()
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.show_image(self.dir_path, self.image_list[self.current_image_index])

    def show_next_image(self):
        if self.save_mode:
            self.export_annotations()
        if self.current_image_index < len(self.image_list) - 1:
            self.current_image_index += 1
            self.show_image(self.dir_path, self.image_list[self.current_image_index])

    def show_image(self, dir_path, image_name):
        self.pixmap = QPixmap(os.path.join(dir_path, image_name))
        self.scene.clear()
        pixmap_item = QGraphicsPixmapItem(self.pixmap)
        self.scene.addItem(pixmap_item)
        rect = self.scene.itemsBoundingRect()
        rect.adjust(-10, -10, 10, 10)  # adjust the bounding rectangle to add space around the image
        self.view.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)
        self.is_new_picture = True
        self.draw_annotations()  # Draw annotations after showing a new image
        self.view.setFocus()  # Set focus on the QGraphicsView instance

    def list_item_clicked(self, item):
        if self.save_mode:
            self.export_annotations()
        image_name = item.text()
        self.current_image_index = self.image_list.index(image_name)  # Update the current image index
        self.show_image(self.dir_path, image_name)

    def classes_item_clicked(self, item):
        # Get the class name from the clicked item
        class_name = item.text()

        # Iterate over all items in the scene
        for scene_item in self.scene.items():
            # Check if the item is a QGraphicsRectItem
            if isinstance(scene_item, QGraphicsRectItem):
                # Create a small rectangle at the top left of the item rectangle
                label_area = QRectF(scene_item.rect().topLeft(), QSizeF(1, 1))
                # Iterate over the items that intersect with this small rectangle
                for sub_item in self.scene.items(label_area):
                    # If the item is a text item and its text matches the class name
                    if isinstance(sub_item, QGraphicsTextItem) and sub_item.toPlainText() == class_name:
                        # If the rectangle is already green, remove the color; otherwise, change it to green
                        if scene_item.brush().color() == QColor("green"):
                            scene_item.setBrush(QBrush(Qt.transparent))  # remove the color
                        else:
                            green_color = QColor("green")
                            green_color.setAlpha(40)  # semi-transparent
                            scene_item.setBrush(QBrush(green_color))
            

    def event(self, event):
        if event.type() == QEvent.Type.Gesture:
            return self.gestureEvent(event)
        return super().event(event)        

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_S:
            self.show_loading_dialog()
            self.export_annotations()
            self.dialog.close()
        if event.key() == Qt.Key.Key_A:
            self.show_prev_image()
        if event.key() == Qt.Key.Key_D:
            self.show_next_image()
        if event.key() == Qt.Key.Key_Z:
            self.zoom_in()
        elif event.key() == Qt.Key.Key_X:
            self.zoom_out()

    def zoom_in(self):
        factor = 1.125
        self.view.setTransform(self.view.transform().scale(factor, factor))

    def zoom_out(self):
        factor = 1 / 1.125
        self.view.setTransform(self.view.transform().scale(factor, factor))

    def export_annotations(self):
        if self.dir_path is None:
            QMessageBox.warning(self, "Warning", "First open a directory")
            return
        self.annotations = []
        for item in self.scene.items():
            if isinstance(item, QGraphicsRectItem):
                rect = item.rect()
                if rect.width() > 5 and rect.height() > 5:  # Only include the annotation if its size is bigger than 5 pixels
                    class_name = ''
                    # Create a small rectangle at the top left of the item rectangle
                    label_area = QRectF(rect.topLeft(), QSizeF(1, 1))
                    # Iterate over the items that intersect with this small rectangle
                    for sub_item in self.scene.items(label_area):
                        if isinstance(sub_item, QGraphicsTextItem):
                            # If the item is a text item, get its text
                            class_name = sub_item.toPlainText()
                            break
                    self.annotations.append({
                        'filename': self.image_list[self.current_image_index],  # Include the name of the image file
                        'x': rect.x(),
                        'y': rect.y(),
                        'width': rect.width(),
                        'height': rect.height(),
                        'class': class_name
                    })

        # Load existing annotations
        try:
            with open('annotations.json', 'r') as f:
                self.existing_annotations = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.existing_annotations = []

        if not self.image_list:
            return
        # Remove existing annotations for the current image
        self.existing_annotations = [a for a in self.existing_annotations if a['filename'] != self.image_list[self.current_image_index]]

        # Append new annotations
        self.existing_annotations.extend(self.annotations)

        # Write all annotations back to the file
        with open('annotations.json', 'w') as f:
            json.dump(self.existing_annotations, f)

    def show_loading_dialog(self):
            # Create a QDialog object
            self.dialog = QDialog(self)
            self.dialog.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
            self.dialog.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)
            self.dialog.setWindowFlag(Qt.WindowType.WindowTransparentForInput, True)
            self.dialog.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

            # Create a QLabel object
            self.loading_label = QLabel(self.dialog)

            # Create a QMovie object
            self.movie = QMovie("/Users/omermersin/Developer/Syhme-Tools/giphy.gif")

            # Set the movie to the QLabel
            self.loading_label.setMovie(self.movie)

            # Start the movie
            self.movie.start()
            layout = QVBoxLayout()
            layout.addWidget(self.loading_label)
            self.dialog.setLayout(layout)
            self.dialog.show()


if __name__ == '__main__':
    app = QApplication([])
    viewer = ImageViewer()
    viewer.show()
    app.exec()