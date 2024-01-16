from flask import Flask, jsonify, request
import ml
import numpy as np
import json 

app = Flask(__name__)

@app.route('/anime', methods=['GET'])
def anime():
    userID = request.args.get('user', '123')
    result = ml.get_recommended_animes_for_user(np.int64(userID))
    return json.dumps(result, indent=2, default=lambda x: str(x) if isinstance(x, np.floating) else None)

# Run the server
if __name__ == '__main__':
    app.run(debug=True)