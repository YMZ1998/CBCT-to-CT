# [CBCT-to-CT](https://github.com/YMZ1998/CBCT-to-CT)

CBCT (Cone Beam Computed Tomography) generates pseudo CT images, which are essential for applications where traditional
CT scans are unavailable or difficult to acquire.

## Dataset

[Synthrad2023](https://synthrad2023.grand-challenge.org/)

[SynthRAD2023 Grand Challenge](https://github.com/SynthRAD2023)

Data structure

```
Brain_Pelvis
-train
|-brain
  |-1BA001
    |-ct.nii.gz
    |-cbct.nii.gz
    |-mask.nii.gz
  |- ...
  |-1BA005
    |-ct.nii.gz
    |-cbct.nii.gz
    |-mask.nii.gz
|-pelvis
  |-1PA001
    |-ct.nii.gz
    |-cbct.nii.gz
    |-mask.nii.gz
  |- ...
  |-1PA004
    |-ct.nii.gz
    |-cbct.nii.gz
    |-mask.nii.gz

-test
|-brain
  |- ...
|-pelvis
  |- ...
```

## Usage

### 1. Data Split

```bash
python data_split.py
```

### 2. Make Dataset

```bash
python make_dataset.py
```

### 3. Train, Test & Predict

```bash
python train.py
python test.py
python predict.py
```

## [Metric](https://github.com/SynthRAD2023/metrics)

## [Preprocess](https://github.com/SynthRAD2023/preprocessing)

## Result

![image](https://github.com/YMZ1998/CBCT-to-CT/blob/main/figure/result.png)

## Environment

```bash
conda env create -f env.yml
```

```bash
conda activate lj_py
```

```bash
conda env export --no-builds > env.yml
```

## PyInstaller Installation Guide:

### 1. Create and Activate Conda Environment

```bash
conda create --name cbct2ct python=3.9
```

```bash
conda activate cbct2ct
```

### 2. Install Required Python Packages

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pyinstaller
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple SimpleITK
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple onnxruntime
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tqdm
```

### 3. Use PyInstaller to Package Python Script

```bash
pyinstaller --name CBCT2CT --onefile --icon=cbct2ct.ico CBCT2CT.py
```

### 4. Clean the Build Cache and Temporary Files

```bash
pyinstaller --clean CBCT2CT.spec
```

### 5. Run the Executable

Once the build is complete, you can run the generated `CBCT2CT.exe` with the required parameters:

```bash
CBCT2CT.exe --cbct_path ./test_data/cbct.nii.gz --mask_path ./test_data/mask.nii.gz --result_path ./result --onnx_path ./checkpoint/cbct2ct.onnx
```

- `--cbct_path`: Path to the input CBCT image file.
- `--mask_path`: Path to the input mask file.
- `--result_path`: Path where the results will be saved.
- `--onnx_path`: Path to the ONNX model.

### 6. Deactivate and Remove Conda Environment

```bash
conda deactivate
conda remove --name cbct2ct --all
```

## Reference

[A Simple Two-stage Residual Network for MR-CT Translation](https://github.com/ZhangZhiHao233/MR-to-CT)
