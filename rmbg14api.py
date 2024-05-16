from flask import Flask, request, send_file
from rembg import new_session, remove
from PIL import Image
import io

app = Flask(__name__)

# 在服务器启动时加载模型
model_name = "rmbg14"
session = new_session(model_name)

@app.route('/api/remove', methods=['POST'])
def remove_background():
    # 获取上传的文件
    file = request.files['file']
    input_image = Image.open(file.stream)
    
    # 使用已加载的模型会话处理图片
    output_image = remove(input_image, session=session)
    
    # 将处理后的图片转换为字节流
    img_byte_arr = io.BytesIO()
    output_image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    return send_file(img_byte_arr, mimetype='image/png')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7009)
