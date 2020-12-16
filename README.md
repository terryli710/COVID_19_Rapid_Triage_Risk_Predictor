# **COVID-19 Outcome Prediction Guideline**

<a href="https://zenodo.org/badge/latestdoi/280965640"><img src="https://zenodo.org/badge/280965640.svg" alt="DOI"></a>

## Environment Building

### Python Environment

Install a python **3.6** environment using **`conda`**

### Packages

1. Install `gdcm` using command 

```shell
conda install -c conda-forge gdcm
```

2. Install majority of packages using command `pip install -r requirements.txt`

3. Install `pytorch` using command conda install 

```
pytorch-cpu==1.1.0 torchvision-cpu==0.3.0 cpuonly -c pytorch
```

Or follow this [link](https://pytorch.org/get-started/previous-versions/#conda-4) to install `pytorch 1.1.0`.

4. Install `pyradiomics` from source using command 

```shell
git clone git://github.com/Radiomics/pyradiomics
```

For unix like systems (MacOSX, linux):

```shell
cd pyradiomics

python -m pip install -r requirements.txt

python setup.py install

python setup.py build_ext --inplace
```

For Windows:

```shell
cd pyradiomics

python -m pip install -r requirements.txt

python setup.py install
```

5. (HELPER) If you run into problem described by this [link](https://github.com/Radiomics/pyradiomics/issues/592) when extracting features, you can replace the files in 

```shell
anaconda\envs\<env_name>\lib\site-packages\radioimcs\
```

 with the corresponding files we provided in folder radiomics_patch.

## Feature extraction pipeline usage

### Input

1. Place unsegmented dicom image series in a folder (e.g. `<dcm>`), and arrange the dicom series in the following way:

```shell
<dcm>
├───<patient_id>
│   └───<study_uid>
│       └───<series_uid>
│           ├───000001.dcm
│    		├───000002.dcm
│    		├───...
│    		└───000333.dcm
├───<patient_id>
│...
```



2. Convert segmentation mask images into PNG format, named them in a sequence of     numbers (e.g. `000.png`, `001.png`, …, `332.png`), and place them in another folder (e.g. `<seg>`) for segmentation, and arrange them in the following way:

```shell
<seg>
├───<patient_id>
│   └───<study_uid>
│       └───<series_uid>
│           ├───000.png
│    		├───001.png
│    		├───...
│    		└───332.png
├───<patient_id>
│...
```

### Execution

1. Open terminal in root directory of the prediction pipeline.

2. Activate the conda environment in the terminal.

3. Extract radiomics features using the following command:

```shell
python proc_radiomic_feature.py --dicom_root <dcm> --lesion_mask_root <seg> --save_root <save_root>
```



### Output

1. Extracted features will be saved in the `<save_root>` directory with file name: “`final_merge_feature.csv`”.

## Prediction Pipeline Usage

### Input

1. For radiomics-only models (`Radiom`), use command 

```shell
python COVID-19_prediction.py --radiomics_data <save_root\final_merge_feature.csv>
```

1. For models that include clinical symptoms, demographics and lab test results     (`RadioClinLab`), you can prepare an additional CSV input, according to “`example_lab_input.csv`” (refer to the “`units.txt`” for units and meanings of the entries). This file should also be placed into the root directory of the prediction pipeline (`<project_root>\<lab_input.csv>`).
2. To use the “`RadioClinLab`” model, use the command line code

```shell
python COVID-19_prediction --radiomics_data <save_root\final_merge_feature.csv> --lab_data <lab_input.csv>
```

