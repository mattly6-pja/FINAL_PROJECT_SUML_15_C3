<i>This project uses Python 3.11.</i>

To run, after cloning:  
1. Create a virtual environment:  
   `python3.11 -m venv .venv`  
2. Use created environment:  
     _If on Linux:_  
       `./.venv/Scripts/activate`  
     _If on Windows:_  
       `.\.venv\Scripts\activate.bat`
3. Proceed to diabetes-predictor folder:  
   `cd diabetes-predictor`   
4. Install required packages:  
   `pip install -r requirements.txt`  
5. Run kedro:  
   `kedro run`  
6. You can also see pipeline visualisation by using below command:  
   `kedro viz`  
7. After kedro is finished, you can run the app via streamlit:  
   `streamlit run streamlit_app.py`  

Exploratory data analysis is available in [eda.html](/diabetes-predictor/docs/eda_diabetes.html)

Metrics for the models are available in mertrics file at: /diabetes-predictor/data/08_reporting/best_models_metrics.csv
