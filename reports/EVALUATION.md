# Evaluation scope

`validation_history.csv` is the original Ultralytics epoch-level results file from the SHWD + Roboflow fine-tuning run. The validation set had 1,886 prepared images. Its best recorded mAP@0.5:0.95 row was epoch 36:

| Epoch | Overall box precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
| ---: | ---: | ---: | ---: | ---: |
| 36 | 0.94507 | 0.93647 | 0.96854 | 0.61751 |

The run stopped after epoch 51. `TRAINING_STATUS.md` records that **no held-out SHWD or Roboflow test evaluation was run** for this fine-tuned checkpoint. No independent webcam, laptop-speed, or Raspberry Pi benchmark is claimed. Validation performance may overstate performance on new camera scenes; see the README limitations.
