import flask
from flask import Flask, render_template, request
from lib.preprocessing import *

val_dict = {'Gender': {'NaN': 1, 'male': 1, 'female': 0},
            'Married': {'NaN': 1, 'yes': 1, 'no': 0},
            'Dependents': {'NaN': 1},
            'Education': {'NaN': 1, 'graduated': 1, 'not graduated': 0},
            'Self_Employed': {'NaN': 0, 'yes': 1, 'no': 0},
            'ApplicantIncome': {'NaN': 4000},
            'CoapplicantIncome': {'NaN': 0},
            'LoanAmount': {'NaN': 10000},
            'Loan_Amount_Term': {'NaN': 360},
            'Credit_History': {'NaN': 1, 'yes': 1, 'no': 0},
            'Property_Area': {'NaN': 0, 'urban': 2, 'semi': 1, 'rural': 0},
            'TotalIncome': {'NaN': 4000}
            }

app = Flask(__name__)


@app.route('/')
def entry():
    return render_template('form.html', the_title='Loan Approval Prediction', form_title='Enter profile details')


@app.route('/analyse', methods=['GET', 'POST'])
def proc():
    test_val = create_df()
    name = request.form["name"]
    amount = request.form["LoanAmount"]
    for feature in test_val.columns:
        if(feature != "TotalIncome"):
            if request.form[feature]:
                if((feature == "ApplicantIncome") or (feature == "CoapplicantIncome") or (feature == "LoanAmount")):
                    test_val[feature] = str(int(request.form[feature])//70)
                else:
                    test_val[feature] = request.form[feature]
            else:
                test_val[feature] = 'NaN'
    fix_missing(test_val, val_dict)
    test_val["TotalIncome"] = test_val["ApplicantIncome"] + \
        test_val["CoapplicantIncome"]
    test_val = test_val.astype(float)
#    scaler = load_scaler()
#    test_val = scaler.transform(test_val)
    resp = {}
    model = load_model()
    pred = model.predict(test_val)
    resp['prediction'] = str(pred).strip('[]')
    status = pred
    return render_template('result.html', status=status, name=name, amount=amount)


if __name__ == '__main__':
    app.run(debug=True)
