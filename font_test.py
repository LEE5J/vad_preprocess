import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import os

# NanumGothic 폰트의 경로 (시스템에 설치된 위치에 맞게 수정)
font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'NanumGothic'
else:
    print("NanumGothic 폰트를 찾을 수 없습니다. 해당 폰트를 설치하거나 경로를 확인해 주세요.")