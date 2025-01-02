import matplotlib.font_manager as fm
for font in fm.findSystemFonts(fontpaths=None, fontext='ttf'):
    if 'Nanum' in font:
        print(font)

# import matplotlib as mpl
# mpl.font_manager._rebuild()

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 한글 폰트 설정
font_path = 'C:/Users/min13/AppData/Local/Microsoft/Windows/Fonts/NanumGothic.ttf'
font_prop = fm.FontProperties(fname=font_path)
plt.rc('font', family=font_prop.get_name())

print(f"Font set to: {font_prop.get_name()}")

# 테스트 플롯
plt.title("테스트: 한글이 제대로 표시되나요?", fontsize=16)
plt.show()


import matplotlib as mpl
print(mpl.get_cachedir())
