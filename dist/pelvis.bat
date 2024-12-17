@echo off

CBCT2CT.exe --cbct_path ./test_data/pelvis/cbct.nii.gz --mask_path ./test_data/pelvis/mask.nii.gz --result_path ./result --anatomy pelvis

pause
