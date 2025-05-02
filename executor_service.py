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

        # Increase buffer size for large outputs
        sys.stdout = buffer
        sys.stderr = buffer

        # Execute with larger timeout
        exec(code, {}, exec_locals)
        
        # Get output
        output = buffer.getvalue()
        result_var = exec_locals.get("result")
        
        if result_var is not None:
            output += str(result_var)

        return_dict["result"] = output or "No output."
        return_dict["success"] = True
        
    except Exception as e:
        return_dict["result"] = f"❌ Error: {str(e)}"
        return_dict["success"] = False
    finally:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

@app.route("/execute", methods=["POST"])
def execute():
    content = request.json
    code = content.get("code", "")
    
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    
    # Increased timeout to 30 seconds
    p = multiprocessing.Process(target=run_code, args=(code, return_dict))
    p.start()
    p.join(30)  # timeout after 30 seconds

    if p.is_alive():
        p.terminate()
        return jsonify({
            "result": "Execution timed out after 30 seconds.",
            "success": False
        })

    return jsonify({
        "result": return_dict["result"],
        "success": return_dict.get("success", False)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9090)