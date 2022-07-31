import cv2
import numpy as np
from calibration.mark_points_on_image import MarkPoints
from image_transformation.transform import Transform


class TransformCameraView:
    def __init__(self, camera_view_image, top_view_image):
        self.camera_image = camera_view_image
        self.top_view_size = (360, 360, 3)
#        self.top_view = np.zeros(self.top_view_size, np.uint8)
        self.top_view = top_view_image
        self.top_view_size = top_view_image.shape
        self.camera_view_points = None
        self.top_view_points = None
        self.transformation_matrix = None
        self.transformed_image = None
        self.data = {}
        self.boxsize = 20

    def mark_points_on_camera_view_image(self, num_points):
        mark_points = MarkPoints(self.camera_image, "Camera View")
        self.camera_view_points = mark_points.mark_points(num_points)

    def mark_points_on_top_view(self, num_points):
        mark_points = MarkPoints(self.top_view, "Top View")
        #bit of a gross way of doing it but works
        
#        center_point = self.top_view_size[0]/2
#        centre_point = mark_points.mark_points(num_points)
#        self.data['points'].append([center_point+self.boxsize,center_point+self.boxsize])
#        self.data['points'].append([center_point-self.boxsize,center_point+self.boxsize])
#        self.data['points'].append([center_point-self.boxsize,center_point-self.boxsize])
#        self.data['points'].append([center_point+self.boxsize,center_point-self.boxsize])
#        self.top_view_points = np.vstack(self.data['points']).astype(float)
        self.top_view_points = mark_points.mark_points(num_points)

    def calculate_transformation(self):
        transform = Transform()
        transformation = transform.calculate_transform(self.camera_view_points, self.top_view_points)
        self.transformation_matrix = transformation[0]

    def generate_transformed_image(self):
        self.transformed_image = cv2.warpPerspective(self.camera_image, self.transformation_matrix,
                                                     self.top_view_size[0:2])
                                                     
    def display_camera_image_on_top_view(self):
        cv2.imshow("Transformed Image", self.transformed_image)
        cv2.waitKey(0)
