from pathlib import Path
from hloc import extract_features, match_features, reconstruction, visualization, pairs_from_retrieval, triangulation
from pycolmap import CameraMode, ImageReaderOptions, IncrementalMapperOptions
from hloc.utils import viz_3d, viz
from hloc.utils.read_write_model import read_model
import pycolmap
import matplotlib.pyplot as plt


def main_function_v1():
    # images = Path('/home/bob/Data/NHR/basketball/img/0')
    # images = Path('/home/bob/VolcapProjects/Hierarchical-Localization/datasets/images/piano2_mapping')
    images = Path('/home/bob/VolcapProjects/Hierarchical-Localization/datasets/images/mafia1_mapping')

    # outputs = Path('outputs/baller/')
    # outputs = Path('outputs/piano/')
    outputs = Path('outputs/mafia1/')

    sfm_pairs = outputs / 'pairs-netvlad.txt'
    sfm_dir = outputs / 'sfm'
    sfm_undistorted = sfm_dir / 'undistorted'
    sfm_transforms = sfm_dir / 'undistorted' / 'transforms.json'

    retrieval_conf = extract_features.confs['netvlad']
    feature_conf = extract_features.confs['disk']
    matcher_conf = match_features.confs['disk+lightglue']

    feature_conf["model"]["max_keypoints"] = 2000
    feature_conf["preprocessing"]["resize_max"] = 1024
    # feature_conf["preprocessing"]["grayscale"] = True

    retrieval_path = extract_features.main(retrieval_conf, images, outputs)
    pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched=7)

    feature_path = extract_features.main(feature_conf, images, outputs)
    match_path = match_features.main(matcher_conf, sfm_pairs, feature_conf['output'], outputs)

    image_options = ImageReaderOptions()
    # image_options.camera_model = "OPENCV"
    mapper_options = IncrementalMapperOptions()
    # mapper_options.max_num_models = 1

    # if Path(input_dir).exists():
    #     # try read in camera model
    #     cameras_data, images_data, points_data = read_model(input_dir, '.bin')
    #     model = pycolmap.Reconstruction()
    #
    #     # 1st
    #     # model.read(sfm_dir)
    #
    #     # 2nd
    #     # Add cameras to the model
    #     for camera_id, camera in cameras_data.items():
    #         model.add_camera(pycolmap.Camera(
    #             id=camera_id,
    #             model=camera.model,
    #             width=camera.width, height=camera.height,
    #             params=camera.params))
    #
    #     # Add images to the model
    #     for image_id, image in images_data.items():
    #         model.add_image(pycolmap.Image(
    #             id=image_id,
    #             camera_id=image.camera_id,
    #             name=image.name,
    #             keypoints=[],
    #             tvec=image.tvec,
    #             qvec=image.qvec
    #         ))
    #
    #     # optional
    #     # Add points to the model
    #     # for point_id, point in points_data.items():
    #     #     model.add_point3D(pycolmap.Point3D(
    #     #         id=point_id,
    #     #         xyz=point.xyz,
    #     #         color=point.rgb,
    #     #         error=point.error
    #     #     ))
    #     #     print("")
    #
    #     # pycolmap.incremental_mapping()
    #
    #     sfm_dir.mkdir(parents=True, exist_ok=True)
    #     new_db = sfm_dir / 'db.db'
    #     triangulation.create_db_from_model(model, new_db)
    #     reconstruction.import_images(images, new_db, CameraMode.AUTO)
    #     image_ids = reconstruction.get_image_ids(new_db)
    #     reconstruction.import_features(image_ids, new_db, feature_path)
    #     reconstruction.import_matches(image_ids, new_db, sfm_pairs, match_path, min_match_score=None,
    #                                   skip_geometric_verification=False)
    #     reconstruction.estimation_and_geometric_verification(new_db, sfm_pairs, True)
    #     reconstruction.estimation_and_geometric_verification(new_db, sfm_pairs, True)
    #     # model = reconstruction.run_reconstruction(sfm_dir, new_db, images, True)
    #     # model = pycolmap.incremental_mapping(new_db, images, outputs)[0]
    #     pycolmap.triangulate_points(model, new_db, images, outputs, True, mapper_options)
    #
    #     visualization.visualize_sfm_2d(model, images, color_by='visibility', n=5)
    #     visualization.visualize_sfm_2d(model, images, color_by='track_length', n=5)
    #     visualization.visualize_sfm_2d(model, images, color_by='depth', n=5)
    #     plt.show()
    #     return

    model = reconstruction.main(sfm_dir, images, sfm_pairs, feature_path, match_path,
                                verbose=True,
                                camera_mode=CameraMode.AUTO,
                                image_options=image_options,
                                mapper_options={
                                    "max_num_models": 1,
                                    "ignore_watermarks": True,
                                    "min_num_matches": 20
                                })

    fig = viz_3d.init_figure()
    viz_3d.plot_reconstruction(fig, model)
    fig.plotly_update()
    fig.show()

    visualization.visualize_sfm_2d(model, images, color_by='visibility', n=5)
    visualization.visualize_sfm_2d(model, images, color_by='track_length', n=5)
    visualization.visualize_sfm_2d(model, images, color_by='depth', n=5)
    plt.show()


if __name__ == "__main__":
    main_function_v1()
