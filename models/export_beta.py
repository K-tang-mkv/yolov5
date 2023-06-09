import torch
import torch.nn as nn
from torch.utils.mobile_optimizer import optimize_for_mobile
import argparse
from pathlib import Path
import platform

import models
from utils.activations import Hardswish, SiLU
from models.experimental import attempt_load
from utils.general import colorstr, check_img_size, check_requirements, file_size, check_version, check_yaml, check_dataset
from utils.dataloaders import LoadImages

MACOS = platform.system() == 'Darwin'  # macOS environment

def parse_opt():
    parser = argparse.ArgumentParser() # define a argument parser
    # add argument into the parser
    parser.add_argument("--weights", type=str, default="../yolopose.pt", help="model need to be export")
    parser.add_argument("--data", type=str, default="data/coco_kpts.yaml", help=" ")
    parser.add_argument("--img-size", type=int, default=[192,192], help="input size need to be exported")
    parser.add_argument("--device", default="cpu", help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument("--include", default=["tflite"], nargs="+", help='torchscript, onnx, openvino, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle')
    parser.add_argument("--export-nms", default="store_true", help='export the nms part in ONNX model')
    # parse arguments
    opt = parser.parse_args()
    print(opt)
    return opt

def export_onnx(weights, model, model_nms, img, output_names, export_nms=False, dynamic=False, simplify=False):
    prefix = colorstr('ONNX:')
    try:
        import onnx

        print(f'{prefix} starting export with onnx {onnx.__version__}...')
        f = weights.replace('.pt', '.onnx')  # filename
        if export_nms:
            torch.onnx.export(model_nms, img, f, verbose=False, opset_version=11, input_names=['images'], output_names=output_names,
                              dynamic_axes={'images': {0: 'batch', 2: 'height', 3: 'width'},  # size(1,3,640,640)
                                            'output': {0: 'batch', 2: 'y', 3: 'x'}} if dynamic else None)
        else:
            torch.onnx.export(model, img, f, verbose=False, opset_version=11, input_names=['images'], output_names=output_names,
                              dynamic_axes={'images': {0: 'batch', 2: 'height', 3: 'width'},  # size(1,3,640,640)
                                            'output': {0: 'batch', 2: 'y', 3: 'x'}} if dynamic else None)


        # Checks
        model_onnx = onnx.load(f)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model
        # print(onnx.helper.printable_graph(model_onnx.graph))  # print

        # Simplify
        if simplify:
            try:
                check_requirements(['onnx-simplifier'])
                import onnxsim

                print(f'{prefix} simplifying with onnx-simplifier {onnxsim.__version__}...')
                model_onnx, check = onnxsim.simplify(model_onnx,
                                                     dynamic_input_shape=dynamic,
                                                     input_shapes={'images': list(img.shape)} if dynamic else None)
                assert check, 'assert check failed'
                onnx.save(model_onnx, f)
            except Exception as e:
                print(f'{prefix} simplifier failure: {e}')
        print(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
    except Exception as e:
        print(f'{prefix} export failure: {e}')

def export_saved_model(model,
                       im,
                       file,
                       dynamic,
                       tf_nms=False,
                       agnostic_nms=False,
                       topk_per_class=100,
                       topk_all=100,
                       iou_thres=0.45,
                       conf_thres=0.25,
                       keras=False,
                       prefix=colorstr('TensorFlow SavedModel:')):
    # YOLOv5 TensorFlow SavedModel export
    try:
        import tensorflow as tf
    except Exception:
        check_requirements(f"tensorflow{'' if torch.cuda.is_available() else '-macos' if MACOS else '-cpu'}")
        import tensorflow as tf
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

    from models.tf import TFModel

    print(f'\n{prefix} starting export with tensorflow {tf.__version__}...')
    f = str(file).replace('.pt', '_saved_model')
    batch_size, ch, *imgsz = list(im.shape)  # BCHW

    tf_model = TFModel(cfg=model.yaml, model=model, nc=model.nc, imgsz=imgsz)
    im = tf.zeros((batch_size, *imgsz, ch))  # BHWC order for TensorFlow
    _ = tf_model.predict(im, tf_nms, agnostic_nms, topk_per_class, topk_all, iou_thres, conf_thres)
    inputs = tf.keras.Input(shape=(*imgsz, ch), batch_size=None if dynamic else batch_size)
    outputs = tf_model.predict(inputs, tf_nms, agnostic_nms, topk_per_class, topk_all, iou_thres, conf_thres)
    keras_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    keras_model.trainable = False
    keras_model.summary()
    if keras:
        keras_model.save(f, save_format='tf')
    else:
        spec = tf.TensorSpec(keras_model.inputs[0].shape, keras_model.inputs[0].dtype)
        m = tf.function(lambda x: keras_model(x))  # full model
        m = m.get_concrete_function(spec)
        frozen_func = convert_variables_to_constants_v2(m)
        tfm = tf.Module()
        tfm.__call__ = tf.function(lambda x: frozen_func(x)[:4] if tf_nms else frozen_func(x), [spec])
        tfm.__call__(im)
        tf.saved_model.save(tfm,
                            f,
                            options=tf.saved_model.SaveOptions(experimental_custom_gradients=False) if check_version(
                                tf.__version__, '2.6') else tf.saved_model.SaveOptions())
    return f, keras_model

def export_tflite(keras_model, im, file, int8, data, nms, agnostic_nms, prefix=colorstr('TensorFlow Lite:')):
    # YOLOv5 TensorFlow Lite export
    import tensorflow as tf

    print(f'\n{prefix} starting export with tensorflow {tf.__version__}...')
    batch_size, ch, *imgsz = list(im.shape)  # BCHW
    f = str(file).replace('.pt', '-fp16.tflite')

    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.target_spec.supported_types = [tf.float16]
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if int8:
        from models.tf import representative_dataset_gen
        dataset = LoadImages(check_dataset(check_yaml(data))['train'], img_size=imgsz, auto=False)
        converter.representative_dataset = lambda: representative_dataset_gen(dataset, ncalib=100)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.target_spec.supported_types = []
        converter.inference_input_type = tf.uint8  # or tf.int8
        converter.inference_output_type = tf.uint8  # or tf.int8
        converter.experimental_new_quantizer = True
        f = str(file).replace('.pt', '-int8.tflite')
    if nms or agnostic_nms:
        converter.target_spec.supported_ops.append(tf.lite.OpsSet.SELECT_TF_OPS)

    tflite_model = converter.convert()
    open(f, "wb").write(tflite_model)
    return f, None

def run(weights, data, img_size=[192,192], device="cpu", include="onnx", export_nms=False):
    img_size *= 2 if len(img_size) == 1 else 1
    include = [x.lower() for x in include]
    pt_file = Path(weights)

    model = attempt_load(pt_file, device)
    labels = model.names

    # Checks
    gs = int(max(model.stride))  # grid size (max stride)
    # opt.img_size = [check_img_size(x, gs) for x in opt.img_size]  # verify img_size are gs-multiples

    # Input
    img = torch.zeros(1, 3, *img_size).to(device)  # image size(1,3,320,192) iDetection

    # Update model
    for k, m in model.named_modules():
        m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
        if isinstance(m, models.common.Conv):  # assign export-friendly activations
            if isinstance(m.act, nn.Hardswish):
                m.act = Hardswish()
            elif isinstance(m.act, nn.SiLU):
                m.act = SiLU()
    model.model[-1].export = True

    for _ in range(2):
        y = model(img) # dry runs, warm cpu

    output_names = None
    if export_nms:
        nms = models.common.NMS(conf=0.01, kpt_label=True)
        nms_export = models.common.NMS_Export(conf=0.01, kpt_label=True)
        y_export = nms_export(y)
        y = nms(y)
        # assert (torch.sum(torch.abs(y_export[0]-y[0]))<1e-6)
        model_nms = torch.nn.Sequential(model, nms_export)
        model_nms.eval()
        output_names = ['detections']

    print(f"\n{colorstr('PyTorch:')} starting from {weights} ({file_size(weights):.1f} MB)")

    # TorchScript export -----------------------------------------------------------------------------------------------
    prefix = colorstr('TorchScript:')
    try:
        print(f'\n{prefix} starting export with torch {torch.__version__}...')
        f = weights.replace('.pt', '.torchscript.pt')  # filename
        ts = torch.jit.trace(model, img, strict=False)
        ts = optimize_for_mobile(ts)  # https://pytorch.org/tutorials/recipes/script_optimized.html
        ts.save(f)
        print(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
    except Exception as e:
        print(f'{prefix} export failure: {e}')

    # export onnx or tflite --------------------------------------------------------------------------------------------
    if "onnx" in include:
        export_onnx(weights, model, model_nms, img, output_names)
    elif "tflite" in include:
        f, s_model = export_saved_model(model.cpu(),
                                           img,
                                           weights,
                                           dynamic=False)
        export_tflite(s_model, img, weights, float, data=data, nms=False, agnostic_nms=False)
    else:
        print(colorstr("Error!!!!!!!!!!!, no correct model format be set up, pls include onnx or tflite"))


if __name__ == "__main__":
    opt = parse_opt()
    run(**vars(opt))
