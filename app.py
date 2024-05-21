from flask import Flask,request,render_template
from src.logger import logging
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

# @app.route('/')
# def index():
#     return render_template('index.html') 

@app.route('/', methods=['GET','POST'])

def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            wckts=float(request.form.get('wckts')),
            Area=float(request.form.get('Area')),
            Pitch=request.form.get('Pitch')
        )

        pred_df=data.get_data_as_data_frame()
        logging.info("New Data Point inside app.py\n")
        logging.info(pred_df)
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        # print("Mid Prediction")

        results=predict_pipeline.predict(pred_df)
        print(f"After Prediction {results}")

        result = round(results[0])

        Pitch=request.form.get('Pitch')

        if (Pitch == 'bat'):
            true_result = result + 10
        else:
            true_result = result

        return render_template('home.html', true_results = true_result)
    

if __name__=="__main__":
    app.run(host="0.0.0.0")        