import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
filename= "pickle_model.pkl"

with open(filename, 'rb') as file:
    count_vect, sgd_clf = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    #int_features = [int(x) for x in request.form.values()]
    #final_features = [np.array(int_features)]
    X_test_new = [x for x in request.form.values()]
    X_test_new_count = count_vect.transform(X_test_new)
    #print(sgd_clf.predict(X_test_new_count))

    prediction = sgd_clf.predict(X_test_new_count)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)