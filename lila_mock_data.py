#!/usr/bin/env python3

"""Generate mock time-domain LILA signals with PyCBC and lunarsky.

This script:
  1. Generates barycentric hp/hc using pycbc.waveform.get_td_waveform.
  2. Computes the lunar detector barycentric position as a function of detector time.
  3. Computes detector-time -> barycentric-time delay:
         t_bary(t_det) = t_det + delay(t_det)
  4. Interpolates hp/hc at the required barycentric times.
  5. Computes a time-dependent long-wavelength antenna pattern.
  6. Produces detector strain:
         h_det(t) = Fp(t) hp[t_bary(t)] + Fc(t) hc[t_bary(t)]
  7. Saves hp, hc, shifted hp/hc, Fp/Fc, delay, and h_det to HDF5.

This is a geometric long-wavelength LILA mock-data model.
It does not include lunar elastic/normal-mode response or LILA noise.
"""

import argparse
import logging

import h5py
import numpy as np

from astropy import constants
import astropy.units as u
from astropy.coordinates import (
    BarycentricTrueEcliptic,
    CartesianRepresentation,
    ICRS,
    Latitude,
    Longitude,
    SkyCoord,
)
from astropy.coordinates.matrix_utilities import rotation_matrix
from astropy.time import Time

from scipy.interpolate import CubicSpline


LOGGER = logging.getLogger(__name__)


# ------------------------------------------------------------
# Basic lunar surface point
# ------------------------------------------------------------

class SurfacePoint:
    """
    Point fixed on the lunar surface.

    Parameters
    ----------
    det_lat_rad : float
        Lunar latitude in radians.
    det_lon_rad : float
        Lunar longitude in radians.
    det_h_m : float
        Height above lunar surface in meters.
    """

    def __init__(self, det_lat_rad, det_lon_rad, det_h_m=0.0):
        self.lat = Latitude(det_lat_rad * u.rad)
        self.lon = Longitude(det_lon_rad * u.rad)
        self.h = det_h_m * u.m


# ------------------------------------------------------------
# Lunar rotation angle
# ------------------------------------------------------------

def lunar_rotation_angle_since_start(obstime_gps, start_time):
    """
    Simple lunar sidereal phase since observation start.

    This is not a full lunar orientation model.
    It is the simple uniform rotation model used in the shared code.

    Parameters
    ----------
    obstime_gps : float
        GPS time in seconds.
    start_time : astropy.time.Time
        Observation start time.

    Returns
    -------
    angle : astropy Quantity
        Lunar rotation angle in radians.
    """
    lunar_sidereal_period = 27.321661 * 86400.0
    time_utc = Time(obstime_gps, format="gps", scale="utc")
    elapsed_time = (time_utc - start_time).to_value("s")

    angle = 2.0 * np.pi * (
        (elapsed_time % lunar_sidereal_period) / lunar_sidereal_period
    )

    return angle * u.rad


# ------------------------------------------------------------
# Barycentric lunar detector position
# ------------------------------------------------------------

def lunar_surface_position_barycentric(surface_point, obstimes):
    """
    Return barycentric coordinates of a lunar surface point.

    The detector is fixed in the Moon-fixed MCMF frame, but the transform
    to barycentric/ecliptic coordinates is time-dependent. Therefore, when
    obstimes is an array, the same fixed MCMF coordinate must be broadcast
    to every time sample.
    """
    from lunarsky import MCMF, MoonLocation

    location = MoonLocation.from_selenodetic(
        surface_point.lon,
        surface_point.lat,
        surface_point.h,
    )

    # Fixed Moon-body coordinate of the detector site.
    mcmf_xyz = location.mcmf.cartesian.xyz.to(u.m)

    n_time = 1 if obstimes.isscalar else len(obstimes)

    # Ensure shape is (3, N).
    if mcmf_xyz.ndim == 1:
        mcmf_xyz = mcmf_xyz.reshape(3, 1)

    if mcmf_xyz.shape[1] == 1 and n_time > 1:
        mcmf_xyz = np.repeat(mcmf_xyz, n_time, axis=1)

    mcmf_rep = CartesianRepresentation(
        x=mcmf_xyz[0],
        y=mcmf_xyz[1],
        z=mcmf_xyz[2],
    )

    mcmf_skycoords = SkyCoord(
        mcmf_rep,
        frame=MCMF(obstime=obstimes),
    )

    barycentric_coords = mcmf_skycoords.transform_to(
        BarycentricTrueEcliptic(equinox=obstimes)
    )

    xyz_q = barycentric_coords.cartesian.xyz.to(u.m)
    xyz_m = xyz_q.to_value(u.m)

    if xyz_m.shape[0] != 3 and xyz_m.shape[-1] == 3:
        xyz_m = xyz_m.T

    return xyz_m

# ------------------------------------------------------------
# Time delay
# ------------------------------------------------------------

def detector_to_barycenter_delay(obstimes, surface_point, ra_rad, dec_rad, debug=False):
    """
    Compute detector-time perspective delay.

    We evaluate the detector position at known detector times t_det.
    Then compute

        delay(t_det) = k . r_det(t_det) / c,

    where k = -n, n points from SSB to source.

    Then the barycentric waveform time is

        t_bary = t_det + delay(t_det).

    Parameters
    ----------
    obstimes : astropy.time.Time
        Detector sample times.
    surface_point : SurfacePoint
        Lunar detector site.
    ra_rad, dec_rad : float
        Source right ascension and declination in radians.
    debug : bool
        Print debug information.

    Returns
    -------
    delays_s : ndarray
        Delay array in seconds.
    r_arr : ndarray, shape (3, N)
        Detector barycentric positions in meters.
    k_arr : ndarray, shape (3, N) or (3,)
        Propagation vector.
    """
    src = SkyCoord(
        ra=ra_rad * u.rad,
        dec=dec_rad * u.rad,
        frame=ICRS(),
    )

    src_ecl = src.transform_to(BarycentricTrueEcliptic(equinox=obstimes))
    src_cart_rep = src_ecl.represent_as(CartesianRepresentation)

    # n points from SSB to source.
    src_vec = src_cart_rep.xyz / src_cart_rep.norm()

    # k points along GW propagation direction, source -> SSB/detector.
    k_vec = -1.0 * src_vec

    r_arr = lunar_surface_position_barycentric(surface_point, obstimes)

    k_arr = k_vec.to_value(u.dimensionless_unscaled)

    # Dot product over spatial index:
    # proj_m(t) = k(t) . r_det(t)
    proj_m = np.einsum("i...,i...->...", k_arr, r_arr)

    delays_s = (proj_m * u.m / constants.c).to_value(u.s)

    if debug:
        LOGGER.debug("delay min/max [s]: %s %s", np.min(delays_s), np.max(delays_s))
        LOGGER.debug("r_arr shape: %s", r_arr.shape)
        LOGGER.debug("k_arr shape: %s", k_arr.shape)

    return delays_s, r_arr, k_arr


# ------------------------------------------------------------
# Detector tensor / antenna pattern
# ------------------------------------------------------------

def detector_tensor_moon_fixed(
    longitude,
    latitude,
    yangle=0.0,
    xangle=None,
    height=0.0,
    xaltitude=0.0,
    yaltitude=0.0,
):
    """
    Construct a Moon-fixed detector tensor and arm vectors.

    The detector is modeled as a long-wavelength Michelson interferometer with
    two arms fixed in the lunar surface frame.

    Parameters
    ----------
    longitude, latitude : astropy Angle
        Detector lunar longitude and latitude.
    yangle : float
        y-arm azimuth angle in radians.
    xangle : float or None
        x-arm azimuth angle in radians. If None, use yangle + pi/2.
    height : float
        Detector height in meters.
    xaltitude, yaltitude : float
        Arm altitude tilts in radians.

    Returns
    -------
    det : dict
        Dictionary containing response tensor and arm vectors.
    """
    from lunarsky import MoonLocation

    if xangle is None:
        xangle = yangle + np.pi / 2.0

    # Baseline response of one arm pointed in the -X direction.
    resp = np.array(
        [
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )

    rm2 = rotation_matrix(-longitude.radian, "z")
    rm1 = rotation_matrix(-1.0 * (np.pi / 2.0 - latitude.radian), "y")

    resps = []
    vecs = []

    for angle, alt in [(yangle, yaltitude), (xangle, xaltitude)]:
        rm0 = rotation_matrix(angle * u.rad, "z")
        rmN = rotation_matrix(-alt * u.rad, "y")
        rm = rm2 @ rm1 @ rm0 @ rmN

        resps.append(rm @ resp @ rm.T / 2.0)
        vecs.append(rm @ np.array([-1.0, 0.0, 0.0]))

    full_resp = resps[0] - resps[1]

    loc = MoonLocation.from_selenodetic(longitude, latitude, height * u.m)
    loc_arr = np.array([loc.x.value, loc.y.value, loc.z.value])

    return {
        "location": loc_arr,
        "response": np.squeeze(full_resp),
        "xresp": resps[1],
        "yresp": resps[0],
        "xvec": vecs[1],
        "yvec": vecs[0],
        "yangle": yangle,
        "xangle": xangle,
        "height": height,
        "xaltitude": xaltitude,
        "yaltitude": yaltitude,
    }


def _normalize_columns(arr):
    """
    Normalize a 3 x N array column-by-column.
    """
    arr = np.asarray(arr, dtype=np.float64)
    norm = np.linalg.norm(arr, axis=0)
    norm = np.where(norm == 0.0, 1.0, norm)
    return arr / norm


def _as_3_by_n_quantity(xyz_q):
    """Return a Cartesian xyz Quantity as a 3 x N array."""
    xyz_q = xyz_q.to(u.m)
    if xyz_q.ndim == 1:
        xyz_q = xyz_q.reshape(3, 1)
    elif xyz_q.shape[0] != 3 and xyz_q.shape[-1] == 3:
        xyz_q = xyz_q.T
    return xyz_q


def _icrs_cartesian_direction_to_mcmf(vec_icrs, obstimes):
    """
    Transform a pure ICRS Cartesian direction vector into MCMF.

    This explicitly neglects translational parallax. Astropy frame transforms
    are coordinate transforms between frames with different origins, so a
    finite-distance source would include an origin-translation/parallax term.
    To get the pure direction rotation, we transform both the endpoint of a
    short vector and the ICRS origin to MCMF, subtract them, and normalize:

        v_MCMF = MCMF(origin + v_ICRS) - MCMF(origin).

    The subtraction removes the frame-origin translation exactly, leaving only
    the orientation transformation into the Moon-fixed frame.

    Parameters
    ----------
    vec_icrs : array-like, shape (3,)
        Unit Cartesian direction vector in ICRS.
    obstimes : astropy.time.Time
        Observation times. Can be scalar or array-like.

    Returns
    -------
    vec_mcmf : ndarray, shape (3, N)
        Unit direction vector in MCMF at each observation time, with parallax
        neglected.
    """
    from lunarsky import MCMF

    vec_icrs = np.asarray(vec_icrs, dtype=np.float64)
    vec_icrs = vec_icrs / np.linalg.norm(vec_icrs)

    n_time = 1 if obstimes.isscalar else len(obstimes)

    # Any nonzero length works because we subtract the transformed origin and
    # then normalize. Use meters only to give Astropy a concrete distance unit.
    x = np.full(n_time, vec_icrs[0]) * u.m
    y = np.full(n_time, vec_icrs[1]) * u.m
    z = np.full(n_time, vec_icrs[2]) * u.m

    endpoint_icrs = SkyCoord(
        CartesianRepresentation(x=x, y=y, z=z),
        frame=ICRS(),
    )

    origin_icrs = SkyCoord(
        CartesianRepresentation(
            x=np.zeros(n_time) * u.m,
            y=np.zeros(n_time) * u.m,
            z=np.zeros(n_time) * u.m,
        ),
        frame=ICRS(),
    )

    endpoint_mcmf = endpoint_icrs.transform_to(MCMF(obstime=obstimes))
    origin_mcmf = origin_icrs.transform_to(MCMF(obstime=obstimes))

    endpoint_xyz = _as_3_by_n_quantity(endpoint_mcmf.cartesian.xyz)
    origin_xyz = _as_3_by_n_quantity(origin_mcmf.cartesian.xyz)

    xyz = (endpoint_xyz - origin_xyz).to_value(u.m)

    return _normalize_columns(xyz)


def source_basis_icrs_to_mcmf(ra_rad, dec_rad, obstimes, psi_rad=0.0):
    """
    Build the source propagation and polarization basis in the Moon-fixed frame.

    The input RA/Dec are ICRS coordinates. This function constructs the usual
    ICRS sky basis vectors and transforms them into MCMF at each observation
    time.

    Parameters
    ----------
    ra_rad, dec_rad : float
        Source right ascension and declination in radians, interpreted in ICRS.
    obstimes : astropy.time.Time
        Detector sample times.
    psi_rad : float
        Polarization angle in radians.

    Returns
    -------
    n_mcmf : ndarray, shape (3, N)
        Unit vector from detector/SSB toward the source, expressed in MCMF.
    x_pol : ndarray, shape (3, N)
        First polarization basis vector in MCMF.
    y_pol : ndarray, shape (3, N)
        Second polarization basis vector in MCMF.
    """
    ca = np.cos(ra_rad)
    sa = np.sin(ra_rad)
    cd = np.cos(dec_rad)
    sd = np.sin(dec_rad)

    # Unit vector from origin to source in ICRS.
    n_icrs = np.array([cd * ca, cd * sa, sd], dtype=np.float64)

    # Local tangent basis on the sky in ICRS.
    # e_ra points toward increasing RA, e_dec toward increasing Dec.
    e_ra_icrs = np.array([-sa, ca, 0.0], dtype=np.float64)
    e_dec_icrs = np.array([-sd * ca, -sd * sa, cd], dtype=np.float64)

    n_mcmf = _icrs_cartesian_direction_to_mcmf(n_icrs, obstimes)
    e_ra_mcmf = _icrs_cartesian_direction_to_mcmf(e_ra_icrs, obstimes)
    e_dec_mcmf = _icrs_cartesian_direction_to_mcmf(e_dec_icrs, obstimes)

    # Numerical cleanup: enforce an orthonormal triad in MCMF.
    e_ra_mcmf = e_ra_mcmf - n_mcmf * np.sum(n_mcmf * e_ra_mcmf, axis=0)
    e_ra_mcmf = _normalize_columns(e_ra_mcmf)

    e_dec_mcmf = e_dec_mcmf - n_mcmf * np.sum(n_mcmf * e_dec_mcmf, axis=0)
    e_dec_mcmf = e_dec_mcmf - e_ra_mcmf * np.sum(e_ra_mcmf * e_dec_mcmf, axis=0)
    e_dec_mcmf = _normalize_columns(e_dec_mcmf)

    # Apply GW polarization rotation in the tangent plane.
    cpsi = np.cos(psi_rad)
    spsi = np.sin(psi_rad)

    x_pol = cpsi * e_ra_mcmf + spsi * e_dec_mcmf
    y_pol = -spsi * e_ra_mcmf + cpsi * e_dec_mcmf

    return n_mcmf, x_pol, y_pol


def antenna_pattern_lunar(
    det,
    ra_rad,
    dec_rad,
    psi_rad,
    obstime,
    start_time=None,
):
    """
    Lunar antenna pattern using a source-vector transformation into MCMF.

    This replaces the simplified approximation

        gha = lunar_phase - ra

    with the more geometrical operation

        ICRS source direction and polarization basis
            -> MCMF(obstime)
            -> contract with Moon-fixed detector tensor.

    Parameters
    ----------
    det : dict
        Detector response dictionary from detector_tensor_moon_fixed().
    ra_rad, dec_rad, psi_rad : float
        Source RA, Dec, and polarization in radians. RA/Dec are ICRS.
    obstime : astropy.time.Time
        Current detector sample time.
    start_time : astropy.time.Time or None
        Kept only for backward compatibility with the old function signature.
        It is not used by this MCMF implementation.

    Returns
    -------
    fplus, fcross : float
        Long-wavelength antenna factors.
    """
    _, x_pol, y_pol = source_basis_icrs_to_mcmf(
        ra_rad=ra_rad,
        dec_rad=dec_rad,
        obstimes=obstime,
        psi_rad=psi_rad,
    )

    resp = det["response"]

    # x_pol and y_pol have shape (3, 1) for scalar obstime.
    x = x_pol[:, 0]
    y = y_pol[:, 0]

    dx = resp.dot(x)
    dy = resp.dot(y)

    fplus = np.sum(x * dx - y * dy)
    fcross = np.sum(x * dy + y * dx)

    return float(fplus), float(fcross)


def antenna_pattern_series(
    det,
    obstimes,
    ra_rad,
    dec_rad,
    psi_rad,
    start_time=None,
):
    """
    Compute Fp(t), Fc(t) over all detector sample times.

    Unlike the simplified hour-angle implementation, this function transforms the ICRS
    source direction and polarization basis into the Moon-fixed MCMF frame
    at each sample time.
    """
    _, x_pol, y_pol = source_basis_icrs_to_mcmf(
        ra_rad=ra_rad,
        dec_rad=dec_rad,
        obstimes=obstimes,
        psi_rad=psi_rad,
    )

    resp = det["response"]

    dx = resp @ x_pol
    dy = resp @ y_pol

    fp = np.sum(x_pol * dx - y_pol * dy, axis=0)
    fc = np.sum(x_pol * dy + y_pol * dx, axis=0)

    return fp.astype(np.float64), fc.astype(np.float64)


# Optional diagnostic: compare the simplified hour-angle formula against the
# new MCMF transformation for short tests.
def antenna_pattern_lunar_hour_angle(
    det,
    ra_rad,
    dec_rad,
    psi_rad,
    obstime,
    start_time,
):
    """
    Simplified hour-angle antenna pattern retained for validation studies.

    The production path uses antenna_pattern_lunar(), which transforms the
    ICRS source basis into MCMF before contracting with the detector tensor.
    """
    t_gps = float(obstime.gps)
    lunar_phase = lunar_rotation_angle_since_start(t_gps, start_time).to_value(u.rad)
    gha = lunar_phase - ra_rad

    cosgha = np.cos(gha)
    singha = np.sin(gha)
    cosdec = np.cos(dec_rad)
    sindec = np.sin(dec_rad)
    cospsi = np.cos(psi_rad)
    sinpsi = np.sin(psi_rad)

    resp = det["response"]

    x = np.array([
        -cospsi * singha - sinpsi * cosgha * sindec,
        -cospsi * cosgha + sinpsi * singha * sindec,
        sinpsi * cosdec,
    ])

    y = np.array([
        sinpsi * singha - cospsi * cosgha * sindec,
        sinpsi * cosgha + cospsi * singha * sindec,
        cospsi * cosdec,
    ])

    dx = resp.dot(x)
    dy = resp.dot(y)

    fplus = np.sum(x * dx - y * dy)
    fcross = np.sum(x * dy + y * dx)

    return float(fplus), float(fcross)



# ------------------------------------------------------------
# Waveform generation and projection
# ------------------------------------------------------------

def generate_barycentric_waveform(
    mass1,
    mass2,
    distance,
    inclination,
    delta_t,
    f_lower,
    f_final,
    approximant,
):
    """
    Generate barycentric-frame hp/hc using PyCBC.

    Returns arrays and the original PyCBC TimeSeries objects.
    """
    from pycbc.waveform import get_td_waveform

    kwargs = dict(
        approximant=approximant,
        mass1=mass1,
        mass2=mass2,
        f_lower=f_lower,
        delta_t=delta_t,
        inclination=inclination,
        distance=distance,
    )

    if f_final is not None:
        kwargs["f_final"] = f_final

    hp, hc = get_td_waveform(**kwargs)

    t = np.asarray(hp.sample_times, dtype=np.float64)
    hp_arr = np.asarray(hp, dtype=np.float64)
    hc_arr = np.asarray(hc, dtype=np.float64)

    return t, hp_arr, hc_arr, hp, hc


def pad_to_duration_keep_merger_near_end(t, hp, hc, duration, delta_t):
    """
    Pad or crop waveform to a requested duration.

    PyCBC TD waveforms often have merger near t=0 and start_time < 0.
    This function returns a time grid from 0 to duration and places
    the generated waveform at the end of that grid.

    This is useful for a year-equivalent signal where the earlier part is
    allowed to be zero if the generated waveform is shorter than duration.
    """
    n_target = int(np.round(duration / delta_t))
    n_current = len(hp)

    if n_current >= n_target:
        hp_out = hp[-n_target:]
        hc_out = hc[-n_target:]
    else:
        n_pad = n_target - n_current
        hp_out = np.concatenate([np.zeros(n_pad), hp])
        hc_out = np.concatenate([np.zeros(n_pad), hc])

    t_out = np.arange(n_target, dtype=np.float64) * delta_t

    return t_out, hp_out, hc_out


def project_to_lila(
    t_bary,
    hp_bary,
    hc_bary,
    start_time,
    ra_rad,
    dec_rad,
    psi_rad,
    det_lat_rad,
    det_lon_rad,
    det_h_m,
    yangle_rad,
    xangle_rad,
    chunk_size=200000,
    debug=False,
):
    """
    Project barycentric hp/hc into detector-frame LILA strain.

    Uses detector-time perspective:
        t_query = t_det + delay(t_det)

    Then interpolates:
        hp_shifted(t_det) = hp_bary[t_query]
        hc_shifted(t_det) = hc_bary[t_query]

    and applies:
        h_det = Fp hp_shifted + Fc hc_shifted.

    Returns
    -------
    results : dict
        Contains detector time, shifted polarizations, delay, antenna
        patterns, and final strain.
    """
    delta_t = t_bary[1] - t_bary[0]
    n = len(t_bary)

    surface_point = SurfacePoint(det_lat_rad, det_lon_rad, det_h_m)

    det = detector_tensor_moon_fixed(
        longitude=surface_point.lon,
        latitude=surface_point.lat,
        yangle=yangle_rad,
        xangle=xangle_rad,
        height=det_h_m,
    )

    # Interpolants in barycentric waveform time.
    hp_interp = CubicSpline(t_bary, hp_bary, extrapolate=False)
    hc_interp = CubicSpline(t_bary, hc_bary, extrapolate=False)

    hp_shifted = np.zeros(n, dtype=np.float64)
    hc_shifted = np.zeros(n, dtype=np.float64)
    delay = np.zeros(n, dtype=np.float64)
    fp = np.zeros(n, dtype=np.float64)
    fc = np.zeros(n, dtype=np.float64)

    # Detector time grid is the output grid.
    t_det = t_bary.copy()

    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)

        t_chunk = t_det[start:stop]

        # Astropy observation times corresponding to detector samples.
        obstimes = start_time + t_chunk * u.s

        delay_chunk, _, _ = detector_to_barycenter_delay(
            obstimes=obstimes,
            surface_point=surface_point,
            ra_rad=ra_rad,
            dec_rad=dec_rad,
            debug=False,
        )

        # Detector-time perspective.
        # For each detector sample time, query barycentric waveform at:
        #     t_bary_query = t_det + delay(t_det)
        t_query = t_chunk + delay_chunk

        valid = (t_query >= t_bary[0]) & (t_query <= t_bary[-1])

        hp_tmp = np.zeros_like(t_chunk)
        hc_tmp = np.zeros_like(t_chunk)

        hp_tmp[valid] = hp_interp(t_query[valid])
        hc_tmp[valid] = hc_interp(t_query[valid])

        fp_chunk, fc_chunk = antenna_pattern_series(
            det=det,
            obstimes=obstimes,
            ra_rad=ra_rad,
            dec_rad=dec_rad,
            psi_rad=psi_rad,
            start_time=start_time,
        )

        hp_shifted[start:stop] = hp_tmp
        hc_shifted[start:stop] = hc_tmp
        delay[start:stop] = delay_chunk
        fp[start:stop] = fp_chunk
        fc[start:stop] = fc_chunk

        if debug:
            LOGGER.debug(
                "chunk %s:%s, valid=%s/%s, delay=[%.3f, %.3f] s",
                start,
                stop,
                np.count_nonzero(valid),
                len(valid),
                delay_chunk.min(),
                delay_chunk.max(),
            )

    h_det = fp * hp_shifted + fc * hc_shifted

    return {
        "t_det": t_det,
        "hp_shifted": hp_shifted,
        "hc_shifted": hc_shifted,
        "delay": delay,
        "Fp": fp,
        "Fc": fc,
        "h_det": h_det,
        "detector": det,
    }


# ------------------------------------------------------------
# HDF5 output
# ------------------------------------------------------------

def write_lila_hdf5(
    output,
    t_bary,
    hp_bary,
    hc_bary,
    projection,
    attrs,
):
    """
    Write LILA signal products to HDF5.
    """
    with h5py.File(output, "w") as f:
        g0 = f.create_group("barycenter")
        g0.create_dataset("time", data=t_bary, compression="gzip")
        g0.create_dataset("hp", data=hp_bary, compression="gzip")
        g0.create_dataset("hc", data=hc_bary, compression="gzip")

        g1 = f.create_group("detector")
        g1.create_dataset("time", data=projection["t_det"], compression="gzip")
        g1.create_dataset(
            "hp_shifted",
            data=projection["hp_shifted"],
            compression="gzip",
        )
        g1.create_dataset(
            "hc_shifted",
            data=projection["hc_shifted"],
            compression="gzip",
        )
        g1.create_dataset("delay", data=projection["delay"], compression="gzip")
        g1.create_dataset("Fp", data=projection["Fp"], compression="gzip")
        g1.create_dataset("Fc", data=projection["Fc"], compression="gzip")
        g1.create_dataset("h_det", data=projection["h_det"], compression="gzip")

        for k, v in attrs.items():
            f.attrs[k] = v


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def positive_float(value):
    """Parse a strictly positive floating-point command-line value."""
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def positive_int(value):
    """Parse a strictly positive integer command-line value."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_arg_parser():
    """Build the command-line parser for LILA mock-data generation."""
    parser = argparse.ArgumentParser(
        description="Generate geometric long-wavelength mock data for LILA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--output", default="lila_signal.hdf5")

    # Source parameters.
    parser.add_argument(
        "--mass1",
        type=positive_float,
        default=1000.0,
        help="Solar masses",
    )
    parser.add_argument(
        "--mass2",
        type=positive_float,
        default=1000.0,
        help="Solar masses",
    )
    parser.add_argument("--distance", type=positive_float, default=1000.0, help="Mpc")
    parser.add_argument("--inclination", type=float, default=0.7, help="rad")

    # Time-domain waveform settings.
    parser.add_argument("--sample-rate", type=positive_float, default=4.0, help="Hz")
    parser.add_argument("--duration", type=positive_float, default=86400.0, help="s")
    parser.add_argument("--f-lower", type=positive_float, default=0.05, help="Hz")
    parser.add_argument("--f-final", type=positive_float, default=None, help="Hz")
    parser.add_argument("--approximant", default="TaylorT4")

    # Source sky parameters in radians.
    parser.add_argument("--ra", type=float, default=1.3, help="rad")
    parser.add_argument("--dec", type=float, default=0.4, help="rad")
    parser.add_argument("--psi", type=float, default=0.2, help="rad")

    # Lunar detector location and orientation.
    parser.add_argument("--det-lat", type=float, default=-np.pi / 2, help="rad")
    parser.add_argument("--det-lon", type=float, default=0.0, help="rad")
    parser.add_argument("--det-height", type=float, default=0.0, help="m")

    parser.add_argument("--yangle", type=float, default=0.0, help="rad")
    parser.add_argument("--xangle", type=float, default=None, help="rad")

    # Observation start.
    parser.add_argument(
        "--start-time",
        type=str,
        default="2015-03-17 08:50:00",
        help="UTC string",
    )

    parser.add_argument("--chunk-size", type=positive_int, default=200000)
    parser.add_argument("--debug", action="store_true")

    return parser


def configure_logging(debug=False):
    """Configure process-wide logging for command-line execution."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def validate_args(parser, args):
    """Validate argument relationships that argparse types cannot express."""
    n_samples = int(np.round(args.duration * args.sample_rate))
    if n_samples < 2:
        parser.error("--duration and --sample-rate must produce at least two samples")

    if args.f_final is not None and args.f_final <= args.f_lower:
        parser.error("--f-final must be greater than --f-lower")

    if not (-np.pi / 2.0 <= args.dec <= np.pi / 2.0):
        parser.error("--dec must be between -pi/2 and pi/2 radians")

    if not (-np.pi / 2.0 <= args.det_lat <= np.pi / 2.0):
        parser.error("--det-lat must be between -pi/2 and pi/2 radians")


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)
    configure_logging(args.debug)

    delta_t = 1.0 / args.sample_rate
    start_time = Time(args.start_time, scale="utc")

    if args.xangle is None:
        args.xangle = args.yangle + np.pi / 2.0

    LOGGER.info("Generating barycentric hp/hc with PyCBC")
    t_raw, hp_raw, hc_raw, hp_ts, hc_ts = generate_barycentric_waveform(
        mass1=args.mass1,
        mass2=args.mass2,
        distance=args.distance,
        inclination=args.inclination,
        delta_t=delta_t,
        f_lower=args.f_lower,
        f_final=args.f_final,
        approximant=args.approximant,
    )

    LOGGER.info("Raw waveform length: %s samples", len(hp_raw))
    LOGGER.info("Raw waveform duration: %.3f s", len(hp_raw) * delta_t)
    LOGGER.info("Raw PyCBC start_time: %s", hp_ts.start_time)

    LOGGER.info("Padding/cropping to requested duration")
    t_bary, hp_bary, hc_bary = pad_to_duration_keep_merger_near_end(
        t=t_raw,
        hp=hp_raw,
        hc=hc_raw,
        duration=args.duration,
        delta_t=delta_t,
    )

    LOGGER.info("Final duration: %.3f s", len(hp_bary) * delta_t)
    LOGGER.info("Final samples: %s", len(hp_bary))

    LOGGER.info("Projecting barycentric signal to LILA detector frame")
    projection = project_to_lila(
        t_bary=t_bary,
        hp_bary=hp_bary,
        hc_bary=hc_bary,
        start_time=start_time,
        ra_rad=args.ra,
        dec_rad=args.dec,
        psi_rad=args.psi,
        det_lat_rad=args.det_lat,
        det_lon_rad=args.det_lon,
        det_h_m=args.det_height,
        yangle_rad=args.yangle,
        xangle_rad=args.xangle,
        chunk_size=args.chunk_size,
        debug=args.debug,
    )

    attrs = {
        "description": "LILA geometric long-wavelength mock-data projection",
        "mass1_msun": args.mass1,
        "mass2_msun": args.mass2,
        "distance_mpc": args.distance,
        "inclination_rad": args.inclination,
        "sample_rate_hz": args.sample_rate,
        "delta_t_s": delta_t,
        "duration_s": args.duration,
        "f_lower_hz": args.f_lower,
        "f_final_hz": -1.0 if args.f_final is None else args.f_final,
        "approximant": args.approximant,
        "ra_rad": args.ra,
        "dec_rad": args.dec,
        "psi_rad": args.psi,
        "det_lat_rad": args.det_lat,
        "det_lon_rad": args.det_lon,
        "det_height_m": args.det_height,
        "yangle_rad": args.yangle,
        "xangle_rad": args.xangle,
        "start_time_utc": args.start_time,
        "delay_convention": "t_bary_query = t_det + k dot r_det(t_det) / c, with k=-n",
        "response_model": "long-wavelength Michelson geometric response",
        "limitations": (
            "No LILA noise PSD, no lunar elastic transfer function, "
            "no relativistic clock corrections; antenna pattern transforms "
            "ICRS source basis to MCMF."
        ),
    }

    LOGGER.info("Writing %s", args.output)
    write_lila_hdf5(
        output=args.output,
        t_bary=t_bary,
        hp_bary=hp_bary,
        hc_bary=hc_bary,
        projection=projection,
        attrs=attrs,
    )

    LOGGER.info("Done")
    LOGGER.info(
        "Delay range [s]: %s %s",
        np.min(projection["delay"]),
        np.max(projection["delay"]),
    )
    LOGGER.info(
        "Fp range: %s %s",
        np.min(projection["Fp"]),
        np.max(projection["Fp"]),
    )
    LOGGER.info(
        "Fc range: %s %s",
        np.min(projection["Fc"]),
        np.max(projection["Fc"]),
    )
    LOGGER.info(
        "h_det max abs: %s",
        np.max(np.abs(projection["h_det"])),
    )


if __name__ == "__main__":
    main()
