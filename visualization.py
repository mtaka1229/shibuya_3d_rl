from PIL import Image
import os
from matplotlib.animation import ArtistAnimation
import matplotlib.pyplot as plt

# 画像フォルダのパスと保存するGIFファイル名を指定
folder_path = "/Users/takahiromatsunaga/bledata/test_graph"  # 実際のフォルダパスに置き換えてください
gif_folder = "/Users/takahiromatsunaga/bledata/test_graph_gif"
file_name = "test.gif"
#os.mkdir(gif_folder)

# 画像フォルダ内のPNG画像を取得
image_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.png')])

# GIF画像のフレームを格納するリストを作成
frames = []
for image_file in image_files:
    image_path = os.path.join(folder_path, image_file)
    img = Image.open(image_path)
    frames.append(img)

#gif_path = os.path.join(gif_folder, file_name)
# GIF画像を保存
#frames[0].save(gif_path, format='GIF', append_images=frames[1:], save_all=True, duration=200, loop=0)

fig = plt.figure(figsize = (10,10), facecolor='lightblue')

ani = ArtistAnimation(fig, frames, interval=100)
plt.show()
