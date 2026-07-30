"""Transducer array geometry.

A cylindrical transducer element is described by 12 parameters (one row of the
``infos`` array):

- ``[0:3]`` center position ``(xc, yc, zc)`` in meters,
- ``[3:6]`` unit vector ``e1``, the cylinder axis,
- ``[6:9]`` unit vector ``e2``, orthogonal to ``e1``, pointing through the
  middle of the arc (toward the imaging center),
- ``[9]`` cylinder radius ``R`` (m),
- ``[10]`` half-aperture angle ``theta_max`` (rad): the arc spans
  ``[-theta_max, theta_max]`` about ``e2``,
- ``[11]`` half-height ``h`` of the element (m).

:func:`translation_rotation_system` generates the full set of element
positions for a rotating and translating linear probe.
:func:`discretize_cylindrical_transducers` samples each element surface into
points, as required by the ``points`` mode of :class:`patas.PAT`.
"""

import numpy as np
import scipy


def translation_rotation_system(transducer_radius: float,
                                transducer_height: float,
                                transducer_width: float,
                                transducer_pitch: float,
                                transducer_nbr_elements: int,
                                transducer_wavelength: float,
                                grid_size: float,
                                rotation_axis=np.array([0, 1, 0]),
                                translation_axis=np.array([1, 0, 0])
                                ):
    """Generate the element positions of a rotation + translation scan.

    The probe is a linear array of ``transducer_nbr_elements`` cylindrical
    elements (pitch ``transducer_pitch`` along the y-axis). It is rotated
    around ``rotation_axis`` and translated along ``translation_axis`` to
    cover the imaged volume; the angular positions follow a Hamming
    apodization profile.

    Args:
        transducer_radius: Radius of the cylindrical elements (m).
        transducer_height: Height of the elements in the arc direction (m).
        transducer_width: Width of the elements along the cylinder axis (m).
        transducer_pitch: Center-to-center distance between elements (m).
        transducer_nbr_elements: Number of elements of the probe.
        transducer_wavelength: Central acoustic wavelength ``c / Fc`` (m).
        grid_size: Extent of the simulation grid to cover with translations (m).
        rotation_axis: Unit vector of the rotation axis.
        translation_axis: Unit vector of the translation direction.

    Returns:
        ``(nTrans, 12)`` array of transducer descriptions in the format
        expected by :class:`patas.PAT` (see the module docstring).

    Raises:
        ValueError: If ``rotation_axis`` or ``translation_axis`` is not of
            unit norm.
    """
    if not np.isclose(np.sum(rotation_axis**2), 1.) or not np.isclose(np.sum(translation_axis**2), 1.):
        raise ValueError("Axis of rotation and translation should be of unit norm")

    # motor configuration
    rotMotor_alpha = np.pi/4
    theta_max = np.arctan(transducer_height / (2*transducer_radius))  # half aperture angle of the transducer
    transMotor = transducer_wavelength * (transducer_radius / transducer_height)

    # step in angle computed with an apodization function (Hamming window)
    coeff = 100  # windowing coefficient
    uni_step = theta_max  # step for uniform sampling
    alpha_uni = np.arange(-rotMotor_alpha, rotMotor_alpha, uni_step)
    nb_rotation = alpha_uni.size

    Apo_Win = np.hamming(coeff * nb_rotation)  # spatial windowing function
    I_apo_win = scipy.integrate.cumulative_trapezoid(Apo_Win / (Apo_Win.size - 1))

    nb_translation = int(np.round(nb_rotation * I_apo_win[-1]))  # number of translation
    translation = I_apo_win[-1] / nb_translation

    a = np.ones(nb_translation+1)
    alpha_translation = np.zeros(nb_translation)

    for i in range(nb_translation):
        a[i+1] = np.argwhere(I_apo_win <= ((i+1)*translation))[-1, 0]
        alpha_translation[i] = (a[i+1]+a[i]-1) / coeff * rotMotor_alpha / nb_rotation - rotMotor_alpha

    list_centers = []
    list_e1 = []
    list_e2 = []
    for j1 in range(nb_translation):
        # for each angle
        alpha = alpha_translation[j1]
        step = transMotor * (1/np.cos(alpha))  # translation step

        number_translation_angle = int(np.round(grid_size / step))

        for i_trans in range(number_translation_angle):
            transx = transducer_radius*np.tan(alpha) + (i_trans-1 - (number_translation_angle - 1)/2) * step
            # transducer_nbr_element//2 in each side of 0
            y0 = transducer_pitch*((transducer_nbr_elements//2)-1) + (transducer_pitch-transducer_width)/2 + transducer_width/2
            y0 = -y0  # center is in the middle of height
            for e in range(transducer_nbr_elements):
                list_e1.append(np.array([0, 1, 0]))
                list_e2.append(np.array([np.sin(alpha), 0, np.cos(alpha)]))
                transy = e*transducer_pitch
                list_centers.append(np.array([-transducer_radius*np.sin(alpha)+transx,
                                              y0 + transy,
                                              transducer_radius-transducer_radius*np.cos(alpha)]))

    infos = np.zeros((len(list_centers), 12))
    infos[:, :3] = np.stack(list_centers)
    infos[:, 3:6] = np.stack(list_e1)
    infos[:, 6:9] = np.stack(list_e2)

    infos[:, 9] = transducer_radius
    infos[:, 10] = theta_max
    infos[:, 11] = transducer_width / 2

    return infos


def discretize_cylindrical_transducers(infos, number_points_height, number_points_width):
    """Sample the surface of each cylindrical element into points.

    Used with ``mode='points'`` of :class:`patas.PAT`: each element surface
    is replaced by a regular grid of point sensors with their associated
    surface areas.

    Args:
        infos: ``(nTrans, 12)`` transducer descriptions (see module
            docstring).
        number_points_height: Number of samples along the arc.
        number_points_width: Number of samples along the cylinder axis.

    Returns:
        Tuple ``(points, area)`` where ``points`` has shape
        ``(nTrans, number_points_height * number_points_width, 3)`` and
        ``area`` the matching per-point surface areas.
    """
    points = np.zeros((infos.shape[0], number_points_width*number_points_height, 3))
    area = np.zeros((infos.shape[0], number_points_width*number_points_height))

    for i in range(infos.shape[0]):
        c = infos[i, 0:3]
        R = infos[i, 9]
        e1 = infos[i, 3:6]
        e2 = infos[i, 6:9]
        theta_max = infos[i, 10]
        h = infos[i, 11]

        theta = np.linspace(-theta_max, theta_max, number_points_height+1)
        mid_theta = 0.5*(theta[1:] + theta[:-1])

        t = np.linspace(-h, h, number_points_width+1)
        mid_t = 0.5*(t[1:] + t[:-1])

        length_theta = R*(theta[1:] - theta[:-1])
        length_t = t[1:] - t[:-1]

        Theta, T = np.meshgrid(mid_theta, mid_t)

        trans_points = np.zeros((number_points_height*number_points_width, 3))

        x = R*np.cos(Theta).reshape(-1)
        y = R*np.sin(Theta).reshape(-1)
        z = T.reshape(-1)

        trans_points[:, 0] = x
        trans_points[:, 1] = y
        trans_points[:, 2] = z

        rot_matrix = np.zeros((3, 3))
        rot_matrix[:, 0] = e2
        rot_matrix[:, 1] = np.cross(e1, e2)
        rot_matrix[:, 2] = e1
        rot_matrix = rot_matrix.T

        trans_points = (rot_matrix.T @ trans_points.T).T + c[None]
        a = length_theta[None] * length_t[:, None]

        points[i] = trans_points
        area[i, :] = a.reshape(-1)

    return points, area
