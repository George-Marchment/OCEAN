This is a sample starting kit for the Plankton challenge. 

According to [NRMA15], a link has been found between the quality of water and the presence of plankton in said water. As water quality increases, the abundance and diversity of plankton does as well. The aim of our project is to build a plankton classifier to assess water quality.

We use for this problem the Bering Sea dataset, an in situ plankton dataset published in May 2019 by Kaichang Cheng [Che19]. This datset contains 7 classes with an even spread.

References and credits: 
- [NRMA15] A. Nair, J.K. Reshma, A. Mathew, and A. Ashok. Effect of water quality on phytoplankton abundance in selected ponds of nedumangad block panchayat, kerala. Emer Life Sci Res, 2015.
- [Che19] K. Cheng. Bering sea dataset. https://doi.org/10.6084/m9.figshare.8146283.v3, 2019.

Prerequisites:
Install Anaconda Python 3.6.6 
You may create a conda environment, with is named python3 to match the '!source activate python3' commands in the notebook.

Usage:

(1) If you are a challenge participant:

- The images directory contains some examples of raw data.

- The three files sample_*_submission.zip are sample submissions ready to go!

- The file README.ipynb contains step-by-step instructions on how to create a sample submission for the Plankton challenge. It is in same folder as the README.md.

At the prompt type:
jupyter notebook README.ipynb

Then you can run the notebook cell by cell.

- modify sample_code_submission/model.py to provide a better model

- zip the contents of sample_code_submission (without the directory, but with metadata), or

- download the public_data and run (double check you are running the correct version of python):

  `python ingestion_program/ingestion.py public_data sample_result_submission ingestion_program sample_code_submission`

then zip the contents of sample_result_submission (without the directory).

(2) If you are a challenge organizer and use this starting kit as a template, ensure that:

- you modify README.ipynb to provide a good introduction to the problem and good data visualization

- sample_data is a small data subset carved out the challenge TRAINING data, for practice purposes only (do not compromise real validation or test data)

- the following programs run properly:

    `python ingestion_program/ingestion.py sample_data sample_result_submission ingestion_program sample_code_submission`

    `python scoring_program/score.py sample_data sample_result_submission scoring_output`

- the metric identified by metric.txt in the utilities directory is the metric used both to compute performances in README.ipynb and for the challenge.

- your code also runs within the Codalab docker (inside the docker, python 3.6 is called python3):

	`docker run -it -v `pwd`:/home/aux codalab/codalab-legacy:py3`
	
	`DockerPrompt# cd /home/aux`
	`DockerPrompt# python3 ingestion_program/ingestion.py sample_data sample_result_submission ingestion_program sample_code_submission`
	`DockerPrompt# python3 scoring_program/score.py sample_data sample_result_submission scoring_output`
	`DockerPrompt# exit`