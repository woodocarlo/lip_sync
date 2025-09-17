import os
from flask import Flask, request, jsonify, send_file
from inference import run_inference

app = Flask(__name__)

# Default runtime directory (Render gives you /tmp for temporary storage)
RUNTIME_DIR = "/tmp"
os.makedirs(RUNTIME_DIR, exist_ok=True)


@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "ok", "message": "LipSync API is running"})


@app.route("/infer", methods=["POST"])
def infer():
    """
    Accepts an audio file and runs lip-sync inference.
    Example: curl -X POST -F "file=@test.wav" https://your-app.onrender.com/infer
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Use form-data with key 'file'"}), 400

    file = request.files["file"]
    input_path = os.path.join(RUNTIME_DIR, "input.wav")
    file.save(input_path)

    try:
        output_path = run_inference(input_path)

        if not os.path.exists(output_path):
            return jsonify({"error": "Inference failed, no output generated"}), 500

        # Return the generated video file
        return send_file(output_path, mimetype="video/mp4")

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Render requires the app to listen on 0.0.0.0 and port from $PORT
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
