""" Utils for mujoco_py """


def set_camera(cam, azimuth=None, elevation=None, distance=None, lookat=None):
    """ Sets camera parameters """
    if azimuth:
        cam.azimuth = azimuth
    if elevation:
        cam.elevation = elevation
    if distance:
        cam.distance = distance
    if lookat:
        for i in range(3):
            cam.lookat[i] = lookat[i]