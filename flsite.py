import pickle
import numpy as np
from flask import Flask, render_template, url_for, request, jsonify

app = Flask(__name__)


menu = [{"name": "Лаба 1", "url": "p_knn"},
        {"name": "Лаба 2", "url": "p_lab2"},
        {"name": "Лаба 3", "url": "p_lab3"}]

model1 = pickle.load(open('model/RazmerObuvi', 'rb'))
model2 = pickle.load(open('model2/Dogs.pkl', 'rb'))
loaded_model_Tree = pickle.load(open('model3/Iris_pickle_fileTREE', 'rb'))


@app.route("/")
def index():
    return render_template('index.html', title="Лабораторные работы, выполненные ФИО", menu=menu)


@app.route("/p_knn", methods=['POST', 'GET'])
def f_lab1():
    if request.method == 'GET':
        return render_template('lab1.html', title="Линейная регрессия", menu=menu, class_model='')
    if request.method == 'POST':
        X_new = np.array([[float(request.form['list1']),
                           float(request.form['list2']),
                           int(request.form['list3'])]])
        pred = model1.predict(X_new)[0][0]
        pred_str = "{:.2f}".format(pred)  # Округление до десятичных знаков
        return render_template('lab1.html', title="Линейная регрессия", menu=menu,
                               class_model="Это: " + pred_str)

@app.route("/p_lab2", methods=['POST', 'GET'])
def f_lab2():
    if request.method == 'GET':
        return render_template('lab2.html', title="Логистическая регрессия", menu=menu)
    if request.method == 'POST':
        X_new = np.array([[float(request.form['list1']),
                           float(request.form['list2']),
                           float(request.form['list3'])]])
        pred = model2.predict(X_new)[0]
        pred_str = str(pred)
        if pred_str == 0 :
            a = 'Лабрадор'
            return a
        else:
            a = 'Ретривер'
            return a
        return render_template('lab2.html', title="Логистическая регрессия", menu=menu,
                               class_model="Это: " + a)

@app.route("/p_lab3")
def f_lab3():
    if request.method == 'GET':
        return render_template('lab3.html', title="Дерево решений", menu=menu)
    if request.method == 'POST':
        X_new = np.array([[int(request.form['list1']),
                           int(request.form['list2']),
                           int(request.form['list3']),
                           float(request.form['list4']),
                           float(request.form['list5']),
                           int(request.form['list6']),
                           int(request.form['list7']),
                           int(request.form['list8'])]])
        pred = loaded_model_Tree.predict(X_new)
        gender = gender_dict[pred[0]]
        return render_template('lab3.html', title="Дерево решений", menu=menu,
                               class_model="Это: " + gender)

@app.route('/api', methods=['get'])
def get_sort():
    X_new = np.array([[float(request.args.get('list1')), #http://localhost:5000/api?list1=160&list2=70&list3=1
                       float(request.args.get('list2')),
                       int(request.args.get('list3'))]])
    pred = model1.predict(X_new)[0][0]
    pred_str = "{:.2f}".format(pred)

    return jsonify(sort=pred_str)



if __name__ == "__main__":
    app.run(debug=True)
