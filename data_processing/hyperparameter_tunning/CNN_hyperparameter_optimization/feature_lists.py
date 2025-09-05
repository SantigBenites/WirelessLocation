# ---- Feature presets (single source of truth) -------------------------------
# Order matters — this is the order X will be built in.
DATASET_TO_FEATURE = {
    # Legacy presets
    "wifi_fingerprinting_data_raw": [
        "freind1_rssi",
        "freind2_rssi",
        "freind3_rssi",
    ],
    "wifi_fingerprinting_data_exponential": [
        "freind1_rssi_over_freind2_rssi",
        "freind1_rssi_over_freind3_rssi",
        "freind2_rssi_over_freind1_rssi",
        "freind2_rssi_over_freind3_rssi",
        "freind3_rssi_over_freind1_rssi",
        "freind3_rssi_over_freind2_rssi",
    ],
    "wifi_fingerprinting_data": [  # rssi + ratios
        "freind1_rssi",
        "freind2_rssi",
        "freind3_rssi",
        "freind1_rssi_over_freind2_rssi",
        "freind1_rssi_over_freind3_rssi",
        "freind2_rssi_over_freind1_rssi",
        "freind2_rssi_over_freind3_rssi",
        "freind3_rssi_over_freind1_rssi",
        "freind3_rssi_over_freind2_rssi",
    ],

    # New dataset recommended subset (exactly what you listed)
    "wifi_fingerprinting_data_extra_features": [
        "freind1_rssi_rssi_1m",
        "freind2_rssi_rssi_1m",
        "freind3_rssi_rssi_1m",
        "freind1_rssi_residual",
        "freind2_rssi_residual",
        "freind3_rssi_residual",
        "freind1_rssi",
        "freind2_rssi",
        "freind3_rssi",
        "freind1_rssi_power_over_freind2_rssi",
        "freind1_rssi_power_over_freind3_rssi",
        "freind2_rssi_power_over_freind1_rssi",
        "freind2_rssi_power_over_freind3_rssi",
        "freind3_rssi_power_over_freind1_rssi",
        "freind3_rssi_power_over_freind2_rssi",
        "freind1_rssi_share",
        "freind2_rssi_share",
        "freind3_rssi_share",
        "beta1_log10d",
        "n_est",
        "freind1_rssi_over_freind2_rssi",
        "freind1_rssi_over_freind3_rssi",
        "freind2_rssi_over_freind1_rssi",
        "freind2_rssi_over_freind3_rssi",
        "freind3_rssi_over_freind1_rssi",
        "freind3_rssi_over_freind2_rssi",
    ],

    "wifi_fingerprinting_data_extra_features_no_leak": [
        "freind1_rssi"
        "freind2_rssi"
        "freind3_rssi"
        "freind1_rssi_share",
        "freind2_rssi_share",
        "freind3_rssi_share",
        "freind1_rssi_over_freind2_rssi",
        "freind1_rssi_over_freind3_rssi",
        "freind2_rssi_over_freind1_rssi",
        "freind2_rssi_over_freind3_rssi",
        "freind3_rssi_over_freind1_rssi",
        "freind3_rssi_over_freind2_rssi",
        "freind1_rssi_power_over_freind2_rssi",
        "freind1_rssi_power_over_freind3_rssi",
        "freind2_rssi_power_over_freind1_rssi",
        "freind2_rssi_power_over_freind3_rssi",
        "freind3_rssi_power_over_freind1_rssi",
        "freind3_rssi_power_over_freind2_rssi",
        "freind1_rssi_residual",
        "freind2_rssi_residual",
        "freind3_rssi_residual",
        "freind1_rssi_rssi_1m"
        "freind2_rssi_rssi_1m"
        "freind3_rssi_rssi_1m"
        "ap_intercepts_freind1_rssi",
        "ap_intercepts_freind2_rssi",
        "ap_intercepts_freind3_rssi",
        "beta1_log10d",
        "n_est",
        "xyfree_intercept_freind1_rssi",
        "xyfree_intercept_freind2_rssi",
        "xyfree_intercept_freind3_rssi",
        "xyfree_gamma_total_power_db"
    ],

    "wifi_fingerprinting_data_meters": [
        "freind1_rssi",
        "freind2_rssi",
        "freind3_rssi",
    ]
}
