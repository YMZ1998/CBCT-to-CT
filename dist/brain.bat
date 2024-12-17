@echo off

CBCT2CT.exe --cbct_path ./test_data/brain/cbct.nii.gz --mask_path ./test_data/brain/mask.nii.gz --result_path ./result --anatomy brain

pause
