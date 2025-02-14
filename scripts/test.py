import scripts.foder_util as foder_util

loader = foder_util.LoadFilesFromDirList()
directory_path = r"C:\Users\zhangbaitao\Desktop\ComfyUI-Inspire-Pack"
try:
    folders = loader.load_files(directory_path, image_load_cap=0, start_index=0)
    print("全路径(带后缀名):", folders[0])
    print("全路径(不带后缀名):", folders[1])
    print("文件名(带后缀名):", folders[2])
    print("文件名(不带后缀名):", folders[3])
    print("上一级目录名:", folders[4])
    print("数量:", folders[5])
except FileNotFoundError as e:
    print(e)

