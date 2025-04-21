import supervisely as sly
from dotenv import load_dotenv
import open3d as o3d
import numpy as np
from supervisely.project.project_type import ProjectType
from supervisely.geometry.pointcloud import Pointcloud
from supervisely.pointcloud_annotation.pointcloud_tag_collection import (
    PointcloudTagCollection,
)
from supervisely.pointcloud_annotation.pointcloud_object_collection import (
    PointcloudObjectCollection,
)
import cv2
from ultralytics import YOLO
import os
import src.globals as g


def get_pcd_ids(dataset_ids):
    """
    Get list of point cloud IDs from given list of dataset IDs

    Parameters:
    - dataset_ids: a list of dataset IDs

    Returns:
    - a list of point cloud IDs
    """

    pcd_ids = []
    for dataset_id in dataset_ids:
        dataset_pcd_infos = g.api.pointcloud.get_list(dataset_id)
        dataset_pcd_ids = [info.id for info in dataset_pcd_infos]
        pcd_ids.extend(dataset_pcd_ids)
    return pcd_ids


def load_pcd_data(pcd_id, save_dir="app_data"):
    """
    Get point cloud data as

    Parameters:
    - pcd_id: an ID of point cloud

    Returns:

    - open3d.geometry.PointCloud object
    - a numpy array of shape shape (N, 3) where each row contains [u, v, z] for a point
    """
    # download input point cloud to local storage
    local_pcd_path = os.path.join(save_dir, f"{pcd_id}.pcd")
    g.api.pointcloud.download_path(pcd_id, local_pcd_path)

    # read input point cloud
    pcd = o3d.io.read_point_cloud(local_pcd_path)
    pcd_points = np.asarray(pcd.points)
    return pcd, pcd_points


def load_photo_context_data(pcd_id):
    """
    Loads photo context image data: image, extrinsic and intrinsic matrices

    Parameters:
    - pcd_id: an ID of point cliud

    Returns:

    - a list of dictionaries containing images, extrinsic and intrinsic matrices
    """
    photo_context_data = []
    # get photo context image info
    img_infos = g.api.pointcloud.get_list_related_images(pcd_id)
    for i, img_info in enumerate(img_infos):
        data = {}
        # download related image
        photo_context_img_path = f"app_data/{pcd_id}_{i}.jpg"
        g.api.pointcloud.download_related_image(img_info["id"], photo_context_img_path)
        photo_context_img = sly.image.read(photo_context_img_path)
        # extract extrinsic and intrinsic matrices from image info
        extrinsic_matrix = img_info["meta"]["sensorsData"]["extrinsicMatrix"]
        extrinsic_matrix = np.asarray(extrinsic_matrix).reshape((3, 4))
        intrinsic_matrix = img_info["meta"]["sensorsData"]["intrinsicMatrix"]
        intrinsic_matrix = np.asarray(intrinsic_matrix).reshape((3, 3))

        data["image"] = photo_context_img
        data["extrinsic_matrix"] = extrinsic_matrix
        data["intrinsic_matrix"] = intrinsic_matrix
        photo_context_data.append(data)
    return photo_context_data


def get_2d_masks(model, photo_context_img):
    """
    Runs model on photo context image and extracts 2D masks from model predictions

    Parameters:
    - model: ultralytics.YOLO model
    - photo_context_img: a numpy array containing photo context image

    Returns:

    - a list of numpy arrays containing masks
    - a list of mask class names
    """
    # predict car masks on photo context image
    predictions = model(
        cv2.cvtColor(photo_context_img.copy(), cv2.COLOR_RGB2BGR), retina_masks=True
    )[0]
    class_names = list(model.names.values())
    # extract masks and their class names from yolo predictions
    masks_2d = []
    mask_class_names = []
    if predictions.masks:
        masks = predictions.masks.data
        bboxes = predictions.boxes.data
        for box, mask in zip(bboxes, masks):
            left, top, right, bottom, confidence, cls_index = (
                int(box[0]),
                int(box[1]),
                int(box[2]),
                int(box[3]),
                float(box[4]),
                int(box[5]),
            )
            class_name = class_names[cls_index]
            mask_class_names.append(class_name)
            mask = mask.cpu().numpy()
            masks_2d.append(mask)
    return masks_2d, mask_class_names


def project_3d_to_uvz(P_w_array, K, R, T):
    """
    Project multiple 3D points in world coordinates to 2D image coordinates (u, v)
    and include the depth (z) for each point

    Parameters:
    - P_w_array: A numpy array of shape (N, 3) where N is the number of 3D points
    - K: The intrinsic matrix (3x3)
    - R: The rotation matrix (3x3)
    - T: The translation vector (3x1)

    Returns:
    - A numpy array of shape (N, 3) where each row contains [u, v, z] for a point
    """

    # convert the world points to camera coordinates
    P_c_array = np.dot(P_w_array, R.T) + T  # (N, 3)

    # extract camera coordinates (X_c, Y_c, Z_c)
    X_c, Y_c, Z_c = P_c_array[:, 0], P_c_array[:, 1], P_c_array[:, 2]

    # calculate the 2D projections (u, v) using the intrinsic matrix
    u = (K[0, 0] * X_c / Z_c) + K[0, 2]
    v = (K[1, 1] * Y_c / Z_c) + K[1, 2]

    # stack the results: [u, v, z] for each point
    uvz = np.vstack([u, v, Z_c]).T  # (N, 3)

    return uvz


def get_3d_masks(uvz, pcd, masks_2d, photo_context_img):
    """
    Create 3D segmentation masks from 3D point projections and 2D masks

    Parameters:
    - uvz: a numpy array of shape (N, 3) where each row contains [u, v, z] for a point
    - pcd: an open3d.geometry.PointCloud object containing input point cloud
    - masks_2d: a list of numpy arrays containing image masks
    - photo_context_img: a numpy array containing photo context image

    Returns:
    - A 2D numpy array of shape (N, 3) where each row contains [u, v, z] for a point
    """
    # extract 3d segmentation masks from 3d point projections
    u, v, z = uvz[:, 0], uvz[:, 1], uvz[:, 2]
    masks_3d = []

    for mask in masks_2d:
        # get indexes of projections which are located inside masks
        inside_masks = []
        img_h, img_w, _ = photo_context_img.shape
        for idx in range(len(pcd.points)):
            point = np.array([int(u[idx]), int(v[idx]), int(z[idx])])
            if (
                (point[0] <= 0 or point[0] >= img_w)
                or (point[1] <= 0 or point[1] >= img_h)
                or point[2] < 0
            ):
                continue
            else:
                if np.all(mask[point[1], point[0]] == 1):
                    inside_masks.append(idx)

        masked_pcd = pcd.select_by_index(inside_masks)
        cluster_labels = np.array(masked_pcd.cluster_dbscan(eps=1.5, min_points=100))
        clusters, counts = np.unique(cluster_labels, return_counts=True)

        biggest_cluster = clusters[np.argsort(counts)][-1:]
        biggest_cluster_indexes = []
        for idx, label in enumerate(cluster_labels):
            if label in biggest_cluster:
                biggest_cluster_indexes.append(idx)
        inside_masks_processed = [inside_masks[idx] for idx in biggest_cluster_indexes]
        masks_3d.append(inside_masks_processed)
    return masks_3d


def upload_masks(pcd_id, masks_3d, mask_class_names, project_id, project_meta):
    """
    Upload 3D segmentation masks to the platform

    Parameters:
    - pcd_id: an ID of point cloud
    - masks_3d: a list of lists containing indexes of point cloud points to be segmented
    - mask_class_names: a list of strings containing mask class names
    - project_meta: sly.ProjectMeta object with point cloud metadata
    """
    pcd_objects = []
    pcd_figures = []

    for cls_name, mask_3d in zip(mask_class_names, masks_3d):
        if not project_meta.get_obj_class(cls_name):
            project_meta = project_meta.add_obj_class(
                sly.ObjClass(cls_name, Pointcloud)
            )
            g.api.project.update_meta(project_id, project_meta.to_json())
        geometry = Pointcloud(mask_3d)
        pcd_object = sly.PointcloudObject(project_meta.get_obj_class(cls_name))
        pcd_figure = sly.PointcloudFigure(pcd_object, geometry)
        pcd_objects.append(pcd_object)
        pcd_figures.append(pcd_figure)

    pcd_objects = PointcloudObjectCollection(pcd_objects)
    result_ann = sly.PointcloudAnnotation(
        pcd_objects, pcd_figures, PointcloudTagCollection([])
    )
    g.api.pointcloud.annotation.append(pcd_id, result_ann)
