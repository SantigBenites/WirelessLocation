from validate_single_model import validate_pt_model_across_groups

MODEL_ROOT = "/home/admindi/sbenites/WirelessLocation/data_processing/hyperparameter_tunning/CNN_hyperparameter_optimization/"

res = validate_pt_model_across_groups(
    model_path=f"{MODEL_ROOT}/model_storage_extra_features_run_no_leak_xy_free/outdoor_indoor_and_garage_run3_depth9_model5.pt",
    db_name="wifi_fingerprinting_data_extra_features_no_leak",  # from your TrainingConfig default
)
print("OVERALL:", res["overall"])
for g, m in res["per_group"].items():
    print(g, m)


for model in 