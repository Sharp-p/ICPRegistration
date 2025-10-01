import os
import open3d as o3d
import numpy as np
import copy
import sys
from pathlib import Path

def main() -> None:
    if len(sys.argv) != 3 and len(sys.argv) != 1:
        print("[ERROR]: correct usage:\t python registration.py <path-to-2dPointCloudTarget.pts> <path-to-2dPointCloudSource\n"
              "or to use predefined 3d Data-Set:\t python registration.py")
        exit(1)
    print(sys.argv)

    target_path = ''
    source_path = ''

    use_dataset = False
    if len(sys.argv) != 3: use_dataset = True
    else:
        target_path = sys.argv[1]
        source_path = sys.argv[2]

    src_path = ''
    trg_path = ''
    if not use_dataset and os.path.exists(target_path) and os.path.exists(source_path):
        src_path, trg_path = generate_pts(target_path, source_path)
    else:
        use_dataset = True
        print("[WARNING]: using dataset")

    threshold = 1

    voxel_size = 0.05  # means 5cm for this dataset
    #generate the down sampled point clouds for the RANSAC algorithm
    source, target, source_down, target_down, source_fpfh, target_fpfh, trans_init = prepare_dataset(
        voxel_size, src_path, trg_path, use_dataset)

    print("Initial alignment")
    print("Threshold={}:".format(threshold))
    evaluation = o3d.pipelines.registration.evaluate_registration(
        source, target, threshold, trans_init)
    print(evaluation)

    #result_ransac = execute_global_registration(source_down, target_down,
    #                                            source_fpfh, target_fpfh,
    #                                            voxel_size)
    #print(result_ransac)
    #draw_registration_result(source_down, target_down, result_ransac.transformation)

    # TODO: switch between noisy and normal data
    mu, sigma = 0, 0.1  # mean and standard deviation
    source_noisy = source

    #apply_noise(source, mu, sigma))

    # TODO: switch between point to point and point to plane
    print("Apply point-to-point ICP")
    # k should match the standard deviation of the noise model of the input datas (obviously not easy with real world data)
    loss = o3d.pipelines.registration.TukeyLoss(k=sigma)
    print("Using robust loss:", loss)
    # alternative use point to place
    # TODO: switch between point to plane and point to point
    tf_est = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_noisy, target, threshold, trans_init, tf_est)

    print(reg_p2p)
    print("Transformation is:")
    print(reg_p2p.transformation)
    draw_registration_result(source, target, reg_p2p.transformation)

def generate_pts(target_path, source_path) -> (str, str):
    src_path = parse_csv(source_path)
    trg_path = parse_csv(target_path)
    return src_path, trg_path

def parse_csv(path_csv: str) -> str:
    # this will break if you change the position of this file
    root = Path(__file__).parent
    split_csv = path_csv.split(os.sep)
    pts_path = os.path.join(root, 'data', f'{split_csv[-1][:-4]}.xyz')

    with open(path_csv) as csv_file:
        lines = csv_file.readlines()

        with open(pts_path, 'w') as pts_file:
            i = 0
            for row in lines:
                split_row = row.split(',')
                if i == 0:
                    i += 1
                    continue
                pts_file.write(f'{split_row[2]} {split_row[3][:-2]} 0\n')
                i += 1
    return pts_path


def draw_registration_result(source, target, transformation) -> None:
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])

def apply_noise(pcd, mu, sigma):
    noisy_pcd = copy.deepcopy(pcd)
    points = np.asarray(noisy_pcd.points)
    points += np.random.normal(mu, sigma, size=points.shape)
    noisy_pcd.points = o3d.utility.Vector3dVector(points)
    return noisy_pcd

def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def prepare_dataset(voxel_size, src_path, trg_path, use_dataset):
    print(":: Load two point clouds and disturb initial pose.")
    # if path setted reads from local data
    if not use_dataset:
        source = o3d.io.read_point_cloud(src_path)
        target = o3d.io.read_point_cloud(trg_path)
        #source.estimate_normals(
            #search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
        #target.estimate_normals(
            #search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))

        print(source)
        print(target)
    else:
        demo_icp_pcds = o3d.data.DemoICPPointClouds()
        source = o3d.io.read_point_cloud(demo_icp_pcds.paths[0])
        target = o3d.io.read_point_cloud(demo_icp_pcds.paths[1])

    trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    source.transform(trans_init)
    draw_registration_result(source, target, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh, trans_init

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result

if __name__ == "__main__":
    main()

