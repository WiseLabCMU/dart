# Hyperparameter Tuning Notes

`alpha_clip` (`--clip`): 0.05 seems to work pretty well.
- If the velocity is very noisy / bad, high `alpha_clip` can cause problems.
- Low `alpha_clip` causes phantom "trails" behind the trajectory of high reflectance + low transmittance, which probably indicates overfitting.

`epochs` (`--epochs`): 3 seems to be enough.
- With an appropriate `alpha_clip`, the val loss doesn't seem to move by much after the first epoch.
- Too many epochs can cause overfitting which is qualitatively apparent on a visual inspection (the loss doesn't really tell the whole story). The reflectance map starts to get a lot of holes, and the transmittance map is even less continuous.
