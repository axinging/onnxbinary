import onnxruntime as ort
import numpy as np

def buildAndRunBinaryGraph():
    # Create the model and check
    size = 3
    inputShape = [size]
    np.random.seed(0)
    a = np.int64(10)
    b = np.int64(20)
    c = np.int64(30)
    input = np.array([a, b, c], dtype='int64').reshape(inputShape)
    MODEL_NAME = './jets-text-to-speech.onnx'
    ort_sess = ort.InferenceSession(MODEL_NAME)
    outputs = ort_sess.run(None, {'text': input})
    print('after run')
    print(outputs[0].shape)
    ol = outputs[0][0].flatten().tolist()
    print(outputs)

buildAndRunBinaryGraph()
