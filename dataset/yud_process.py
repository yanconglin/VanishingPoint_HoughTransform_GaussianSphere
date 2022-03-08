import glob
import os
import numpy as np
import scipy.io
import imageio
import matplotlib.pyplot as plt
# import lsd.lsd as lsd
import csv
import copy
import argparse
import skimage.io as sio

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


class YUDVP:

    def __init__(self, data_dir_path="./data", split='all', keep_in_memory=True, normalize_coords=False,
                 return_images=False, extract_lines=False, yudplus=True):
        self.data_dir = data_dir_path
        self.lines_dir = os.path.join(self.data_dir, 'lines')
        self.vps_dir = os.path.join(self.data_dir, 'vps')
        self.orig_dir = os.path.join(self.data_dir, 'YorkUrbanDB')

        self.lines_files = glob.glob(self.lines_dir + "/P*.txt")
        self.lines_files.sort()
        self.image_ids = [os.path.splitext(os.path.basename(x))[0] for x in self.lines_files]
        self.vps_files = [os.path.join(self.vps_dir, x + "GroundTruthVP_CamParams.mat") for x in self.image_ids]
        self.image_files = [os.path.join(self.orig_dir, "%s/%s.jpg" % (x, x)) for x in self.image_ids]

        self.keep_in_mem = keep_in_memory
        self.normalize_coords = normalize_coords
        self.return_images = return_images
        self.extract_lines = extract_lines
        self.yudplus = yudplus

        if split is not None:
            if split == "train":
                self.set_ids = list(range(0, 25))
            elif split == "test":
                self.set_ids = list(range(25, 102))
            elif split == "all":
                self.set_ids = list(range(0, 102))
            else:
                assert False, "invalid split"

        self.dataset = [None for _ in self.set_ids]

        camera_params = scipy.io.loadmat(os.path.join(self.orig_dir, "cameraParameters.mat"))

        f = camera_params['focal'][0, 0]
        ps = camera_params['pixelSize'][0, 0]
        pp = camera_params['pp'][0, :]

        self.K = np.matrix([[f / ps, 0, pp[0]], [0, f / ps, pp[1]], [0, 0, 1]])
        self.S = np.matrix([[2.0 / 640, 0, -1], [0, 2.0 / 640, -0.75], [0, 0, 1]])
        self.K_inv = np.linalg.inv(self.S * self.K) if normalize_coords else np.linalg.inv(self.K)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, key):

        id = self.set_ids[key]
        datum = self.dataset[key]

        if datum is None:

            image_path = self.image_files[id]
            mat_gt_path = self.vps_files[id]
            lines_path = self.lines_files[id]
            print('datum', image_path, mat_gt_path, lines_path)
            image_rgb = imageio.imread(image_path)
            image = rgb2gray(image_rgb)

            if self.extract_lines:
                lsd_line_segments = lsd.detect_line_segments(image)
            else:
                lsd_line_segments = []
                with open(lines_path, 'r') as csv_file:
                    reader = csv.DictReader(csv_file, delimiter=' ')
                    for line in reader:
                        p1x = float(line['point1_x'])
                        p1y = float(line['point1_y'])
                        p2x = float(line['point2_x'])
                        p2y = float(line['point2_y'])
                        lsd_line_segments += [np.array([p1x, p1y, p2x, p2y])]
                lsd_line_segments = np.vstack(lsd_line_segments)

            if self.normalize_coords:
                lsd_line_segments[:, 0] -= 320
                lsd_line_segments[:, 2] -= 320
                lsd_line_segments[:, 1] -= 240
                lsd_line_segments[:, 3] -= 240
                lsd_line_segments[:, 0:4] /= 320.

            line_segments = np.zeros((lsd_line_segments.shape[0], 12))
            for li in range(line_segments.shape[0]):
                p1 = np.array([lsd_line_segments[li, 0], lsd_line_segments[li, 1], 1])
                p2 = np.array([lsd_line_segments[li, 2], lsd_line_segments[li, 3], 1])
                centroid = 0.5 * (p1 + p2)
                line = np.cross(p1, p2)
                line /= np.linalg.norm(line[0:2])
                line_segments[li, 0:3] = p1
                line_segments[li, 3:6] = p2
                line_segments[li, 6:9] = line
                line_segments[li, 9:12] = centroid

            gt_data = scipy.io.loadmat(mat_gt_path)
            true_vds = np.matrix(gt_data['vp'])
            true_vds[1, :] *= -1

            true_vps = self.K * true_vds

            num_vp = true_vps.shape[1] if self.yudplus else 3
            tvp_list = []

            for vi in range(num_vp):
                true_vps[:, vi] /= true_vps[2, vi]

                tVP = np.array(true_vps[:, vi])[:, 0]
                tVP /= tVP[2]

                tvp_list += [tVP]

            true_vps = np.vstack(tvp_list)

            vps = true_vps

            datum = {'line_segments': line_segments, 'lsd_line_segments': lsd_line_segments, 'VPs': vps, 'id': id, 'VDs': true_vds, 'image': image_rgb}

            if self.return_images:
                datum['image'] = np.array(image_rgb)

            for vi in range(datum['VPs'].shape[0]):
                if self.normalize_coords:
                    datum['VPs'][vi, :][0] -= 320
                    datum['VPs'][vi, :][1] -= 240
                    datum['VPs'][vi, :][0:2] /= 320.
                    datum['VPs'][vi, :] /= np.linalg.norm(datum['VPs'][vi, :])

            if self.keep_in_mem:
                self.dataset[key] = datum

        return datum


def line_vp_distances(lines, vps):
    distances = np.zeros((lines.shape[0], vps.shape[0]))

    for li in range(lines.shape[0]):
        for vi in range(vps.shape[0]):
            vp = vps[vi, :]
            line = lines[li, 6:9]
            centroid = lines[li, 9:12]
            constrained_line = np.cross(vp, centroid)
            constrained_line /= np.linalg.norm(constrained_line[0:2])

            distance = 1 - np.abs((line[0:2] * constrained_line[0:2]).sum(axis=0))

            distances[li, vi] = distance
    return distances

def to_pixel(vpts, focal_length=1.0, h=480, w=640):
    x = vpts[:,0] / vpts[:, 2] * focal_length * max(h, w)/2.0 + w//2
    y = -vpts[:,1] / vpts[:, 2] * focal_length * max(h, w)/2.0 + h//2
    return y, x

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='NYU-VP dataset visualisation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', default="dataset/YorkUrbanDB/yud_plus/data", help='where to load')
    parser.add_argument('--save_dir', default='dataset/dataset/YorkUrbanDB/yud_plus/processed_data', help='where to save')
    # parser.add_argument('--data_dir', default="/home/yclin/cluster/dataset/YorkUrbanDB/yud_plus/data", help='where to load')
    # parser.add_argument('--save_dir', default='processed_data', help='where to save')
    opt = parser.parse_args()
    if not os.path.exists(opt.save_dir): os.makedirs(opt.save_dir)
    ### confusing names in the YUDVP class ...
    dataset = YUDVP(data_dir_path=opt.data_dir, split='all', normalize_coords=False, return_images=True, extract_lines=False)

    max_num_vp = 0
    for idx in range(len(dataset)):
        image = dataset[idx]['image']
        vps = dataset[idx]['VPs']
        num_vps = vps.shape[0]
        print("image no. %04d -- vps: %d" % (idx, num_vps))
        if num_vps > max_num_vp: max_num_vp = num_vps

        ls_lsd = dataset[idx]['lsd_line_segments']
        vp = dataset[idx]['VPs']
        vp[:,0] /= vp[:,2]
        vp[:,1] /= vp[:,2]
        vp[:,2] /= vp[:,2]

        ls = dataset[idx]['line_segments']
        distances = line_vp_distances(ls, vp)
        closest_vps = np.argmin(distances, axis=1)
        # print('closest_vps', closest_vps)

        vps_pixel = vp[:, 0:2]
        vps_homo = np.concatenate((vps_pixel, np.ones((len(vps_pixel),1), dtype=np.float32)), axis=-1)
        vps_homo[:, 0] -= 320
        vps_homo[:, 1] -= 240
        vps_homo[:, 1] *= -1
        vps_homo[:, 0:2] /= 320.
        vps_homo /= np.linalg.norm(vps_homo, axis=1, keepdims=True)

        for j in range(0,4):
            if j==0:
                image_name = os.path.join(opt.save_dir, f'{idx:04d}_0.png')
                sio.imsave(image_name, image)
                npz_name = os.path.join(opt.save_dir, f'{idx:04d}_0.npz')
                np.savez_compressed(npz_name, line_segments=ls_lsd, line_vp_inds=closest_vps, vpts=vps_homo)

            if j == 1:  # # # left-right
                image_new= image.copy()[::, ::-1]
                ls_lsd_new = copy.deepcopy(ls_lsd)
                ls_lsd_new[:, 0] *= -1.
                ls_lsd_new[:, 0] += image.shape[1]
                ls_lsd_new[:, 2] *= -1.
                ls_lsd_new[:, 2] += image.shape[1]

                vps_homo_new = copy.deepcopy(vps_homo)
                vps_homo_new[:, 0] *= -1.0

                image_name = os.path.join(opt.save_dir, f'{idx:04d}_1.png')
                sio.imsave(image_name, image_new)
                npz_name = os.path.join(opt.save_dir, f'{idx:04d}_1.npz')
                np.savez_compressed(npz_name, line_segments=ls_lsd_new, line_vp_inds=closest_vps, vpts=vps_homo_new)
            #
            if j == 2:  # # # top-down
                image_new= image.copy()[::-1, ::]
                ls_lsd_new = copy.deepcopy(ls_lsd)
                ls_lsd_new[:, 1] *= -1.
                ls_lsd_new[:, 1] += image.shape[0]
                ls_lsd_new[:, 3] *= -1.
                ls_lsd_new[:, 3] += image.shape[0]

                vps_homo_new = copy.deepcopy(vps_homo)
                vps_homo_new[:, 1] *= -1.0

                image_name = os.path.join(opt.save_dir, f'{idx:04d}_2.png')
                sio.imsave(image_name, image_new)
                npz_name = os.path.join(opt.save_dir, f'{idx:04d}_2.npz')
                np.savez_compressed(npz_name, line_segments=ls_lsd_new, line_vp_inds=closest_vps, vpts=vps_homo_new)

            if j == 3:  # # # top-down and left-right
                image_new = image.copy()[::-1, ::-1]
                ls_lsd_new = copy.deepcopy(ls_lsd)
                ls_lsd_new[:, 1] *= -1.
                ls_lsd_new[:, 1] += image.shape[0]
                ls_lsd_new[:, 3] *= -1.
                ls_lsd_new[:, 3] += image.shape[0]

                ls_lsd_new[:, 0] *= -1.
                ls_lsd_new[:, 0] += image.shape[1]
                ls_lsd_new[:, 2] *= -1.
                ls_lsd_new[:, 2] += image.shape[1]

                vps_homo_new = copy.deepcopy(vps_homo)
                vps_homo_new[:, 1] *= -1.0
                vps_homo_new[:, 0] *= -1.0

                image_name = os.path.join(opt.save_dir, f'{idx:04d}_3.png')
                sio.imsave(image_name, image_new)
                npz_name = os.path.join(opt.save_dir, f'{idx:04d}_3.npz')
                np.savez_compressed(npz_name, line_segments=ls_lsd_new, line_vp_inds=closest_vps, vpts=vps_homo_new)

        # #### visualize  #################
        # colours = ['#e6194b', '#4363d8', '#aaffc3', '#911eb4', '#46f0f0', '#f58231', '#3cb44b', '#f032e6',
        #            '#008080', '#bcf60c', '#fabebe', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3',
        #            '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000']
        #
        # fig = plt.figure(figsize=(16, 5))
        # ax1 = fig.add_subplot(1,1,1)
        # # ax1.imshow(image)
        # ax1.imshow(image_new)
        # # ys, xs = to_pixel(vps_homo)
        # ys, xs = to_pixel(vps_homo_new)
        # for x,y in zip(xs, ys):
        #     plt.scatter(x, y, color='r')
        # for li in range(ls.shape[0]):
        #     vpidx = closest_vps[li]
        #     c = colours[vpidx]
        #     # ax1.plot([ls_lsd[li, 0], ls_lsd[li, 2]], [ls_lsd[li, 1], ls_lsd[li, 3]], c=c, lw=2)
        #     ax1.plot([ls_lsd_new[li, 0], ls_lsd_new[li, 2]], [ls_lsd_new[li, 1], ls_lsd_new[li, 3]], c=c, lw=2)
        #
        # fig.tight_layout()
        # plt.show()
