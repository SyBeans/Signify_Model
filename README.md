To run this environment install on the terminal using these command line
python3 -m venv venv or python -m venv venv based on the version you downloaded inn your machine/local

=== Installation process ===
But the case is there is a conflict between mediapipe and protobuf if the same environment so the solution is
separating them a venv so two venv for this project 

Use this command for extraction virtual environment 
# python3 -m venv venv_extract 
for extracting the landmarks first 

Then install the following dependencies on the requirements_extract.txt using the commandline
# pip install -r requirements_extract.txt

Use this after downloading the extraction to train the extracted landmarks
# python3 -m venv venv_train

Then install the following dependencies on the requirements_train.txt using the commandline
# pip install -r requirements_train.txt


Before you install the dependencies first you need to activate below the steps

To activate the environment 
# source venv_extract/bin/activate  # Linux
# source venv_train/bin/activate
or
# venv_extract\Scripts\activate # Windows
# venv_train\Scripts\activate

To deactivate the environment
# deactivate

=== Activating the Virtaul Environemt ===
Must activate them one by one
activate the venv_extract then run the 
# extract_landmark.py 
for extracting the landmarks from the video
Then deactivate the venv_extract just by command 
# deactivate 
after deactivating it activate the 
# venv_train
then run
# train_model.py 
to train the model 
