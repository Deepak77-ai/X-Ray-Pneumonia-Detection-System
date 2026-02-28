#  Xray Lung Classifier

## Problem statement
Pneumonia is an inflammatory condition of the lung affecting primarily the small air sacs known as alveoli.Symptoms typically include some combination of productive or dry cough, chest pain, fever and difficulty breathing. The severity of the condition is variable. Pneumonia is usually caused by infection with viruses or bacteria and less commonly by other microorganisms, certain medications or conditions such as autoimmune diseases.Risk factors include cystic fibrosis, chronic obstructive pulmonary disease (COPD), asthma, diabetes, heart failure, a history of smoking, a poor ability to cough such as following a stroke and a weak immune system. Diagnosis is often based on symptoms and physical examination. Chest X-ray, blood tests, and culture of the sputum may help confirm the diagnosis.The disease may be classified by where it was acquired, such as community- or hospital-acquired or healthcare-associated pneumonia.
Our task is to create a API whichs predict whether the given images are penumonia or not.

## Solution Proposed
The solution proposed for the above problem is that we have used Computer vision to solve the above problem to classify the data. We have used the Pytorch
framework to solve the above problem also we have have created our custom CNN network with the help of pytorch. 


## Dataset used
The dataset was shared by Apollo diagnostic center for research purpose. 

## Tech Stack Used
1. Python 
2. stream-lit
3. Pytorch
5. AWS


## Infrastructure required
1. AWS S3
2. AWS App Runner
3. Github Actions

## How to run
go to my live application(sometimes it takes few sec to load) -->  https://iv3kva4wxasask33xbpeyc.streamlit.app/

         OR

Step 1. Cloning the repository.

https://github.com/Deepak77-ai/-normal-or-pnumonia-detection-using-X_ray-imeges/
```

Step 2: create an .env file & give the keys mension below

export AWS_ACCESS_KEY_ID=<AWS_ACCESS_KEY_ID>

export AWS_SECRET_ACCESS_KEY=<AWS_SECRET_ACCESS_KEY>

export AWS_DEFAULT_REGION=<AWS_DEFAULT_REGION>


Step 3. Create an environment using : Python -m venv my_env

step 4. activate an evirnoment : my_env\Scripts\activate

Step 5. Install the requirements : pip install -r requirements.txt

Step 6. run an app.py using : streamlit run app.py" or using "python app.py


## Conclusion
- The project we have created can also be in real-life by doctors to check whether the person is having Pneumonia or not. It will help doctors to take
better decisions.
