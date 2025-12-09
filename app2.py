# --- 修正箇所ここから ---

# サイドバー設定
st.sidebar.header("🎨 文字設定")
text_input = st.sidebar.text_input("追加する文字", "")
text_color = st.sidebar.color_picker("文字色", "#FFFFFF")
font_size = st.sidebar.slider("フォントサイズ (px)", 10, 200, 50)
text_position = st.sidebar.selectbox("文字の位置", ["中央", "右下", "左下", "右上", "左上"], index=1)

# フォント選択機能
FONT_DIR = "fonts"  # フォントを入れるフォルダ名

# フォルダが存在しない、または空の場合の処理
available_fonts = []
if os.path.exists(FONT_DIR):
    # .ttf または .otf ファイルだけをリストアップ
    available_fonts = [f for f in os.listdir(FONT_DIR) if f.endswith(('.ttf', '.otf'))]

if available_fonts:
    # フォントが見つかった場合、セレクトボックスを表示
    selected_font_name = st.sidebar.selectbox("フォント選択", available_fonts)
    # 選択されたフォントのフルパスを作成
    current_font_path = os.path.join(FONT_DIR, selected_font_name)
else:
    # フォントがない場合
    st.sidebar.warning(f"⚠️ '{FONT_DIR}' フォルダにフォントファイル(.ttf)がありません。")
    current_font_path = None # Noneだとデフォルトフォントになります

# --- 修正箇所ここまで ---

# (中略：ファイルアップロード部分などはそのまま)

# --- 呼び出し部分の修正 ---
# ループ内の add_text_to_image を呼び出す部分で、
# 固定の変数ではなく current_font_path を渡すようにします。

    # 2. 文字入れ処理
    final_img = add_text_to_image(
        resized_img, 
        text_input, 
        current_font_path,  # <--- ここ変数を変更
        font_size, 
        text_color, 
        text_position
    )