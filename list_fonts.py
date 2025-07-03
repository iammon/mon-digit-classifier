import matplotlib.font_manager as fm

for font in fm.findSystemFonts(fontpaths=None, fontext='ttf'):
    font_props = fm.FontProperties(fname=font)
    name = font_props.get_name()
    if "Myanmar" in name or "Noto" in name:
        print(name)
