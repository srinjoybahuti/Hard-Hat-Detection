# Combined dataset preparation report

## Outcome

Prepared 10,818 training, 1,886 validation, 1,205 preserved SHWD test, and 706 separate Roboflow test images.
The configured `test` split is the original SHWD test set. Roboflow test remains available as `dataset/images/test_roboflow` but is not used during training.

## Classes

Verified Roboflow metadata: `0=head`, `1=helmet`, `2=person`. Mapped `head -> no_helmet (1)`, `helmet -> helmet (0)`, and removed `person` boxes. Images that become label-empty are retained as valid negative examples.

## Validation and leakage

Found 0 non-person data issues and excluded invalid annotations/images as itemized in `dataset_report.json`. Removed 0 cross-split leakage candidates. Final exact SHA-256 overlap counts are: train/val=0, train/SHWD-test=0, val/SHWD-test=0.
Roboflow source IDs (filename before `.rf.<hash>`) were also checked across splits. Source images are preserved; output images are symlinks, and remapped labels are new files.

## Reproducibility

Run `.venv/bin/python prepare_combined.py` only in a fresh experiment directory. Detailed counts, issue categories, duplicate examples, and object totals are in `dataset_report.json`.
