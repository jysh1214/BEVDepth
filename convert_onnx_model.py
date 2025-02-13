"""
python3 convert_onnx_model.py
"""
import onnx
original_model = onnx.load_model("bevdepth.onnx")
converted_model = onnx.version_converter.convert_version(original_model, 17)
onnx.save(converted_model, "bevdepth_17.onnx")
