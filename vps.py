# vanishing points
import vp_detection as vp

import utils as utils


def estimate_vps(img):
    # Find the vanishing points of the image, through Xiaohulu methoed
    length_thresh = 60  # Minimum length of the line in pixels
    seed = None  # Or specify whatever ID you want (integer) Ex: 1337
    vpd = vp.VPDetection(length_thresh=length_thresh, seed=seed)
    vps = vpd.find_vps(img)

    if utils.debug >= 0:
        print("  Vanishing points found")
    if utils.debug > 1:
        print("      vps coordinates:\n", vpd.vps_2D)
    if utils.debug > 2:
        print("      length threshold:", length_thresh)
        print("      principal point:", vpd.principal_point)
        print("      focal length:", vpd.focal_length)
        print("      seed:", seed)

    return vpd.vps_2D
