import supervisely as sly
import os
import src.functions as f
from ultralytics import YOLO
import src.globals as g


def main():
    # get project and dataset ids
    project_id = sly.env.project_id()
    dataset_id = sly.env.dataset_id(raise_not_found=False)
    if dataset_id:
        dataset_ids = [dataset_id]
    else:
        dataset_ids = [
            dataset_info.id for dataset_info in g.api.dataset.get_list(project_id)
        ]
    # get project meta
    project_meta = sly.ProjectMeta.from_json(g.api.project.get_meta(project_id))
    if not os.path.exists("app_data"):
        os.mkdir("app_data")
    # load YOLO model
    model = YOLO("yolo11s-seg.pt")
    # get point cloud ids
    pcd_ids = f.get_pcd_ids(dataset_ids)
    # initialize progress bar
    progress_bar = sly.Progress(
        message="Transfering masks from photo context images to point clouds...",
        total_cnt=len(pcd_ids),
    )
    # process each point cloud
    for pcd_id in pcd_ids:
        # get point cloud data
        pcd, pcd_points = f.load_pcd_data(pcd_id)
        # get photo context images and their metadata
        photo_context_infos = f.load_photo_context_data(pcd_id)
        for i, photo_context_info in enumerate(photo_context_infos):
            photo_context_img = photo_context_info["image"]
            extrinsic_matrix = photo_context_info["extrinsic_matrix"]
            intrinsic_matrix = photo_context_info["intrinsic_matrix"]
            # extract rotation matrix and translation vector from extrinsic matrix
            rotation_matrix = extrinsic_matrix[:, :3]
            translation_vector = extrinsic_matrix[:, 3]
            # get 2d masks
            masks_2d, mask_class_names = f.get_2d_masks(model, photo_context_img)
            if len(masks_2d) > 0:
                # get point projections
                uvz = f.project_3d_to_uvz(
                    pcd_points, intrinsic_matrix, rotation_matrix, translation_vector
                )
                # get 3d masks
                masks_3d = f.get_3d_masks(uvz, pcd, masks_2d, photo_context_img)
                # upload masks
                f.upload_masks(
                    pcd_id, masks_3d, mask_class_names, project_id, project_meta
                )
            else:
                sly.logger.warn(
                    f"No objects found. Point cloud ID: {pcd_id}, photo context index: {i}"
                )
        progress_bar.iter_done_report()


if __name__ == "__main__":
    sly.main_wrapper("main", main)
