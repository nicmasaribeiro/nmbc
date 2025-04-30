from flask import Flask, request, jsonify
import multiprocessing
import sys
import contextlib
import io
from flask_cors import CORS


app = Flask(__name__)

CORS(app, supports_credentials=True)


def run_code(code, return_dict):
    try:
        exec_locals = {}
        buffer = io.StringIO()

        # ✅ Capture all printed output
        with contextlib.redirect_stdout(buffer):
            exec(code, {}, exec_locals)

        # ✅ Combine printed output and variable result
        output = buffer.getvalue()
        result_var = exec_locals.get("result")

        if result_var is not None:
            output += str(result_var)

        return_dict["result"] = output or "No output."
    except Exception as e:
        return_dict["result"] = f"❌ Error: {str(e)}"


@app.route("/execute", methods=["POST"])
def execute():
    content = request.json
    code = content.get("code", "")
    
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    p = multiprocessing.Process(target=run_code, args=(code, return_dict))
    p.start()
    p.join(3)  # timeout after 3 seconds

    if p.is_alive():
        p.terminate()
        return jsonify({"result": "Execution timed out."})

    return jsonify({"result": return_dict["result"]})


@app.route("/execute/two/<code>", methods=["POST"])
def execute_two(code):
    content = request.json
    code = code #content.get("code", "")
    
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    p = multiprocessing.Process(target=run_code, args=(code, return_dict))
    p.start()
    p.join(3)  # timeout after 3 seconds

    if p.is_alive():
        p.terminate()
        return jsonify({"result": "Execution timed out."})

    return jsonify({"result": return_dict["result"]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9090)
