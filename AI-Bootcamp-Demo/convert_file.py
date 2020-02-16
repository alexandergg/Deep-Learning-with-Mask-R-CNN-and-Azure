import cv2, base64, numpy

def base64_image(base64str):
    base64img = base64str.encode('utf-8')
    r = base64.decodestring(base64img)
    numpy_buffer = numpy.frombuffer(r, dtype=numpy.uint8)
    img = cv2.imdecode(numpy_buffer, cv2.IMREAD_COLOR)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return rgb_img

def load_image(image64):
    """
    Load and preprocess an image
    """
    im = base64_image(image64)
    return im