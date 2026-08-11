# Optional reference: FastWAM VISUAL reconcile

Copied from the local LeRobot integration that fixed SR≈0.
Not a full package — paste into your FastWAM processor / factory if you hit the same Hub preprocessor mismatch.

```python
def reconcile_fastwam_processors(config, preprocessor, postprocessor):
    """Align loaded FastWAM processors with the model image contract.

    FastWAM expects images in [0, 1] at the policy input; the Wan VAE boundary
    then applies x * 2 - 1. Some released Hub checkpoints ship VISUAL: MEAN_STD
    with mean=std=0.5 (already [0,1]→[-1,1]). Applying both yields ~[-3,1] and
    collapses LIBERO success rate to ~0%. Force VISUAL to IDENTITY on load.
    """
    for step in preprocessor.steps:
        if not isinstance(step, NormalizerProcessorStep):
            continue
        visual_mode = step.norm_map.get(FeatureType.VISUAL)
        if visual_mode is None or visual_mode == NormalizationMode.IDENTITY:
            continue
        logging.warning(
            "FastWAM: overriding loaded preprocessor VISUAL normalization %s → IDENTITY "
            "so images stay in [0, 1] before the model maps them to [-1, 1] for the Wan VAE.",
            visual_mode,
        )
        step.norm_map[FeatureType.VISUAL] = NormalizationMode.IDENTITY
    return preprocessor, postprocessor
```

Call this after loading pretrained preprocessor/postprocessor for `FastWAMConfig`.
