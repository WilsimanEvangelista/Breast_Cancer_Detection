from flask import Blueprint, render_template, request, redirect, url_for
import os
from models.gradcam_utils import aply_gradcam_binary

main = Blueprint("main", __name__)

@main.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        file = request.files.get("image")
        if file:
            file_path = os.path.join("uploads", file.filename)
            file.save(file_path)

            # Binary Model Inference
            binary_result = "Benign"  # Replace with model prediction

            # GradCam Generation
            predicted_label_binary = aply_gradcam_binary(file_path)

            # Multiclass Model Inference
            multiclass_result = "Type A"  # Replace with model prediction

            return render_template(
                "report.html",
                image=file.filename,
                heatmap=f"heatmap_{file.filename}",
                binary_result=binary_result,
                multiclass_result=multiclass_result,
            )
    return render_template("home.html")
