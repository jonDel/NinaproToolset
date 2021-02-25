# NinaproToolset

## Usage
All the usage examples listed bellow ilustrate how to use Ninapro Toolset.
They are Google Colaboratory jupyter notebooks, so you don't need any 
hardware or software beyond an internet browser to run them.
For executing examples 2 and 3, you will need to register an account on http://ninapro.hevs.ch/.

You must fill the fields nina_user and nina_password in the configuration file
src/data/database/config.yaml with your account user and password, encoded in base 64,
in order to the allow the automatic donwloader to download the required datasets.

Ex: Encoding username in base 64, in python:
```python
>>> from base64 import b64encode
>>> user_encoded = b64encode(b"Jon Doe")
```

## Example 1: Training and plotting confusion matrices for subject 5, database 1
Obs: this particular dataset is already included in the repository, so you don't need to download
anything or register an account.

[Colab link](https://colab.research.google.com/drive/11pj1lQpOcHk8te4F9lR4_8nvosq4qeDo?usp=sharing)

## Example 2: Training all subjects from database 1:

[Colab link](https://colab.research.google.com/drive/1UFoTdDlsRHpSy_f8swH2HTICUWbxmvlx?usp=sharing)

## Example 3: Training all subjects from database 3:

[Colab link](https://colab.research.google.com/drive/1aDTnbARRpKNi2xyiV_TFYJqnjCOrNZAO?usp=sharing)
