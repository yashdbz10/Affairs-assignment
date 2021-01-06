import pickle
from wsgiref import simple_server
from flask import Flask, request, app , render_template
from flask import Response
from flask_cors import CORS
import numpy as np

app = Flask(__name__)

@app.route('/',methods=['GET'])

def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST','GET'])
def predictRoute():
    if request.method == 'POST':
        try:
            int_features = [int(x) for x in request.form.values()]
            final_features= [np.array(int_features)]
            with open("sandardScalar.sav", 'rb') as f:
                scalar = pickle.load(f)
            with open("modelForPrediction.sav", 'rb') as f:
                model = pickle.load(f)
                scaled_data = scalar.transform(final_features)
                predict = model.predict(scaled_data)
            if predict[0] == 1:
                result = 'Affair'
            else:
                result ='No-Affair'
            return render_template('index.html', prediction_text='women is having {}'.format(result))
        except Exception as e:
            print('exception is   ', e)
            return 'something is wrong'
    else:
        return render_template('index.html')
if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8005, debug=True)
    #httpd = simple_server.make_server(host, port, app)
    # print("Serving on %s %d" % (host, port))
    #httpd.serve_forever()
