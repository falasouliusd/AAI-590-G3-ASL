from flask import Flask, request, jsonify
import torch, cv2, numpy as np, base64
from src.inference.pipeline import build_model, preprocess_clip, decode_topk

app = Flask(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"
model, meta = build_model(device=device)

@app.route("/predict", methods=["POST"])
def predict():
    # expects JSON with "frames": list of base64 JPGs OR "video_path"
    data = request.get_json()
    frames = []
    if "frames" in data:
        for b64 in data["frames"]:
            img = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)[:,:,::-1]
            frames.append(img)
    elif "video_path" in data:
        cap = cv2.VideoCapture(data["video_path"])
        while True:
            ok, frame = cap.read()
            if not ok: break
            frames.append(frame[:,:,::-1])
        cap.release()
    else:
        return jsonify({"error":"Provide 'frames' or 'video_path'"}), 400

    clip = preprocess_clip(frames)
    with torch.no_grad():
        logits = model(clip.to(device))
    preds = decode_topk(logits, meta["id2gloss"], k=5)
    return jsonify({"predictions":[{"gloss":g,"prob":float(p)} for g,p in preds]})

if __name__=="__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
