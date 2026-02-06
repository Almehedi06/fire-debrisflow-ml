from __future__ import annotations

import numpy as np


def compute_ksat(pH, clay_total, silt_total, cation_exchange_capacity):
    return 10 ** (
        0.40220
        + (0.26122 * pH)
        + 0.44565
        - (0.02329 * clay_total)
        - (0.01265 * silt_total)
        - (0.01038 * cation_exchange_capacity)
    )


def compute_transmissivity(ksat, soil_thickness, min_value: float = 0.1):
    transmissivity = ksat * soil_thickness
    return np.where(transmissivity < min_value, min_value, transmissivity)


def compute_saturated_water_content(dry_bulk_density, clay_total, silt_total):
    return (
        0.83080
        - (0.28217 * dry_bulk_density)
        + (0.0002728 * clay_total)
        + (0.000187 * silt_total)
    )


def classify_soil_texture(sand, silt, clay):
    if sand >= 85 and (silt + 1.5 * clay) < 15:
        return 1
    if sand >= 70 and sand <= 91 and (silt + 1.5 * clay) >= 15 and (silt + 2 * clay) < 30:
        return 2
    if clay >= 7 and clay <= 20 and sand > 52 and (silt + 2 * clay) >= 30:
        return 3
    if clay < 7 and silt < 50 and sand > 43:
        return 3
    if clay >= 7 and clay <= 27 and silt >= 28 and silt <= 50 and sand <= 52:
        return 4
    if clay >= 12 and clay <= 27 and silt >= 50:
        return 5
    if clay < 12 and silt <= 80 and silt >= 50:
        return 5
    if clay < 12 and silt >= 80:
        return 6
    if clay >= 20 and clay <= 35 and silt < 28 and sand > 45:
        return 7
    if clay >= 27 and clay <= 40 and sand < 46 and sand > 20:
        return 8
    if clay >= 27 and clay <= 40 and sand <= 20:
        return 9
    if clay >= 35 and sand >= 45:
        return 10
    if clay >= 40 and sand >= 40:
        return 11
    if clay >= 40 and sand <= 45 and silt < 40:
        return 12
    return 0


def compute_soil_texture(sand, silt, clay):
    sand = np.asarray(sand)
    silt = np.asarray(silt)
    clay = np.asarray(clay)
    out = np.zeros_like(sand, dtype=float)
    for i in range(len(sand)):
        out[i] = classify_soil_texture(sand[i], silt[i], clay[i])
    return out


def classify_fc_wp(soil_texture, saturated_water_content):
    psi_fc = 3.33  # m
    psi_wp = 350  # m

    if soil_texture == 1:  # USDA - sand; USCS - SP
        phi = 30
        porosity = 0.395
        theta_fc = porosity * ((0.121 / psi_fc) ** (1 / 4.05))
        theta_wp = porosity * ((0.121 / psi_wp) ** (1 / 4.05))
    elif soil_texture == 2:  # USDA - loamy sand; USCS - SM
        phi = 30
        porosity = 0.410
        theta_fc = porosity * ((0.090 / psi_fc) ** (1 / 4.38))
        theta_wp = porosity * ((0.090 / psi_wp) ** (1 / 4.38))
    elif soil_texture == 3:  # USDA - sandy loam; USCS - SM
        phi = 30
        porosity = 0.435
        theta_fc = porosity * ((0.218 / psi_fc) ** (1 / 4.90))
        theta_wp = porosity * ((0.218 / psi_wp) ** (1 / 4.90))
    elif soil_texture == 4:  # USDA - loam; USCS - CL
        phi = 31
        porosity = 0.451
        theta_fc = porosity * ((0.478 / psi_fc) ** (1 / 5.39))
        theta_wp = porosity * ((0.478 / psi_wp) ** (1 / 5.39))
    elif soil_texture == 5:  # USDA - silt loam; USCS - ML
        phi = 31
        porosity = 0.485
        theta_fc = porosity * ((0.786 / psi_fc) ** (1 / 5.30))
        theta_wp = porosity * ((0.786 / psi_wp) ** (1 / 5.30))
    elif soil_texture == 6:  # USDA - silt; USCS - ML
        phi = 31
        porosity = saturated_water_content if saturated_water_content is not None else -9999.0
        theta_fc = 0.2
        theta_wp = 0.075
    elif soil_texture == 7:  # USDA - sandy clay loam; USCS - SC
        phi = 27
        porosity = 0.420
        theta_fc = porosity * ((0.299 / psi_fc) ** (1 / 7.12))
        theta_wp = porosity * ((0.299 / psi_wp) ** (1 / 7.12))
    elif soil_texture == 8:  # USDA - clay loam; USCS - CL
        phi = 16
        porosity = 0.476
        theta_fc = porosity * ((0.630 / psi_fc) ** (1 / 8.52))
        theta_wp = porosity * ((0.630 / psi_wp) ** (1 / 8.52))
    elif soil_texture == 9:  # USDA - silty clay loam; USCS - CL
        phi = 16
        porosity = 0.477
        theta_fc = porosity * ((0.356 / psi_fc) ** (1 / 7.75))
        theta_wp = porosity * ((0.356 / psi_wp) ** (1 / 7.75))
    elif soil_texture == 10:  # USDA - sandy clay; USCS - SC
        phi = 27
        porosity = 0.426
        theta_fc = porosity * ((0.153 / psi_fc) ** (1 / 10.4))
        theta_wp = porosity * ((0.153 / psi_wp) ** (1 / 10.4))
    elif soil_texture == 11:  # USDA - silty clay; USCS - CH
        phi = 16
        porosity = 0.492
        theta_fc = porosity * ((0.490 / psi_fc) ** (1 / 10.4))
        theta_wp = porosity * ((0.490 / psi_wp) ** (1 / 10.4))
    elif soil_texture == 12:  # USDA - clay; USCS - CH
        phi = 16
        porosity = 0.482
        theta_fc = porosity * ((0.405 / psi_fc) ** (1 / 11.4))
        theta_wp = porosity * ((0.405 / psi_wp) ** (1 / 11.4))
    else:
        porosity = -9999.0
        theta_fc = -9999.0
        theta_wp = -9999.0
        phi = -9999.0

    return porosity, theta_fc, theta_wp, phi


def compute_fc_wp_arrays(soil_texture, saturated_water_content):
    soil_texture = np.asarray(soil_texture)
    saturated_water_content = np.asarray(saturated_water_content)

    porosity = np.zeros_like(soil_texture, dtype=float)
    theta_fc = np.zeros_like(soil_texture, dtype=float)
    theta_wp = np.zeros_like(soil_texture, dtype=float)
    phi = np.zeros_like(soil_texture, dtype=float)

    for i in range(len(soil_texture)):
        porosity[i], theta_fc[i], theta_wp[i], phi[i] = classify_fc_wp(
            soil_texture[i],
            saturated_water_content[i],
        )

    return porosity, theta_fc, theta_wp, phi


def compute_soil_density(dry_bulk_density, porosity):
    return (dry_bulk_density / (1 - porosity)) * 1000
