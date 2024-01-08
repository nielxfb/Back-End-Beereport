from flask import Flask, request, jsonify
from flask_cors import CORS
from scipy.spatial import distance
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app, supports_credentials=True)

@app.route('/process-data', methods=['POST'])
def process_data():
    data = request.json
    df = pd.read_csv('./data.csv')
    
    input = {}
    for course_name, course_details in data.items():
        input[course_name] = float(course_details)

    idf = pd.DataFrame(input, index=[0])

    dist = []
    for i in range(0, df.shape[0]-1):
        dist.append([i, distance.euclidean(idf.iloc[0].values, df.iloc[i, 1:].values)])

    dist = np.array(dist)
    dist = dist[dist[:, 1].argsort()]

    top_5_recommendations = []
    for i in range(0, 5):
        top_5_recommendations.append(df.iloc[int(dist[i, 0]), 0])

    return jsonify(top_5_recommendations)


if __name__ == '__main__':
    app.run(debug=True, port=8000)
