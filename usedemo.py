from PIL import Image
from rembg import new_session, remove

input_path = 'input.png'
output_path = 'output.png'

input = Image.open(input_path)
model_name = "u2net"
session = new_session(model_name)
output = remove(input, session=session)
output.save(output_path)
