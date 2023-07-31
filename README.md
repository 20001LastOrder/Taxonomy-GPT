# Prompting or Fine-tuning? A Comparative Study of Large Language Models for Taxonomy Construction

### Set Up
```
git clone git@github.com:20001LastOrder/Taxonomy-GPT.git
cd Taxonomy-GPT
conda create -n Taxonomy-GPT python=3.9
conda activate Taxonomy-GPT
pip install -r requirements.txt
```
### Guidline


#### Step 1: Get Original Input datasets
* Wordnet: We retrieve data from [Catherin et al](https://github.com/cchen23/ctp/tree/master/datasets/data_creators/bansal-taxo-generalsetup) 
* ACM CCS: We get data in a form of OWL ontologies from [ACM Digital Library](https://dl.acm.org/ccs)
You can find all the original inputs from /data/original_input.

#### Step 2: Data Preprocessing

Please check and run: 
ccs_data_preprocessing.ipynb and wordnet_data_preprocessing.ipynb

Below is the Taxonomies splitting steps for ACM CCS dataset:
![split taxonomies to sub-taxonomies](/images/acm_ccs_preprocessing.png)

#### Step 3: Train Model

Check settings in config.yml
Run train.py

#### Step 4: Evaluation

Check evaluation.ipynb
View utils.py for evaluation algorithm 

#### Step 5: Violation Measures

violation.ipynb


