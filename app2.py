import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import io
import os

# --- 関数定義 ---

def smart_resize(img_pil, target_width, target_height):
    """顔認識をしてリサイズする関数"""
    img_np = np.array(img_pil)
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    orig_h, orig_w = img_cv.shape[:2]

    try:
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    except Exception:
        faces = []

    center_x, center_y = orig_w / 2, orig_h / 2
    
    if len(faces) > 0:
        min_x = np.min(faces[:, 0])
        min_y = np.min(faces[:, 1])
        max_x = np.max(faces[:, 0] + faces[:, 2])
        max_y = np.max(faces[:, 1] + faces[:, 3])
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2

    scale = max(target_width / orig_w, target_height / orig_h)
    resized_w, resized_h = int(orig_w * scale), int(orig_h * scale)
    
    img_resized = img_pil.resize((resized_w, resized_h), Image.LANCZOS)
    
    center_x_scaled = center_x * scale
    center_y_scaled = center_y * scale
    left = center_x_scaled - (target_width / 2)
    top = center_y_scaled - (target_height / 2)
    
    left = max(0, min(left, resized_w - target_width))
    top = max(0, min(top, resized_h - target_height))
    
    final_img = img_resized.crop((left, top, left + target_width, top + target_height))
    return final_img

def add_text_to_image(img, text, font_path, font_size, color, position):
    """画像に文字を追加する関数"""
    if not text:
        return img

    img_with_text = img.copy()
    draw = ImageDraw.Draw(img_with_text)
    W, H = img.size

    # フォント設定
    try:
        if font_path and os.path.exists(font_path):
            font = ImageFont.truetype(font_path, font_size)
        else:
            font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()
    
    # テキストサイズ取得
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
    except AttributeError:
        # 古いPillowバージョンの場合のフォールバック
        text_w, text_h = draw.textsize(text, font=font)

    # 位置計算
    x, y = 0, 0
    padding = 20

    if position == "中央":
        x = (W - text_w) / 2
        y = (H - text_h) / 2
    elif position == "左上":
        x = padding
        y = padding
    elif position == "左下":
        x = padding
        y = H - text_h - padding
    elif position == "右上":
        x = W - text_w - padding
        y = padding
    elif position == "右下":
        x = W - text_w - padding
        y = H - text_h - padding

    # 描画
    draw.text((x, y), text, fill=color, font=font)
    return img_with_text


# --- アプリのメイン処理 ---

# 1. ページ設定 (これは必ず一番最初に実行する必要がある)
st.set_page_config(page_title="簡単リサイズ＆文字入れ", layout="wide")
st.title("📷 AI自動リサイズ & 文字入れ")
st.markdown("画像をアップロードすると、人物を中心にトリミングし、文字を追加します。")

# 2. サイドバー設定 (文字やフォントの設定)
st.sidebar.header("🎨 文字設定")
text_input = st.sidebar.text_input("追加する文字", "")
text_color = st.sidebar.color_picker("文字色", "#FFFFFF")
font_size = st.sidebar.slider("フォントサイズ (px)", 10, 200, 50)
text_position = st.sidebar.selectbox("文字の位置", ["中央", "右下", "左下", "右上", "左上"], index=1)

# フォント選択機能
FONT_DIR = "fonts"  # フォントを入れるフォルダ名
current_font_path = None

# フォルダチェックとセレクトボックス表示
if os.path.exists(FONT_DIR):
    available_fonts = [f for f in os.listdir(FONT_DIR) if f.endswith(('.ttf', '.otf'))]
    if available_fonts:
        selected_font_name = st.sidebar.selectbox("フォント選択", available_fonts)
        current_font_path = os.path.join(FONT_DIR, selected_font_name)
    else:
        st.sidebar.warning(f"⚠️ '{FONT_DIR}' フォルダ内には.ttfファイルがありません。")
else:
    # フォルダがない場合は警告せず、デフォルト動作にする（または警告を出しても良い）
    st.sidebar.info(f"💡 プロジェクト内に '{FONT_DIR}' フォルダを作って .ttf ファイルを入れるとフォントを選べます。")


# 3. メインコンテンツ (画像アップロードと処理)
uploaded_file = st.file_uploader("ここに画像をドラッグ＆ドロップ", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # 画像を開く
    image = Image.open(uploaded_file)
    st.image(image, caption="元の画像", width=400)
    st.divider()
    
    st.subheader("👇 変換結果")
    
    # 出力サイズ設定
    targets = [
        (1080, 1080, "正方形 (1:1)"),
        (1920, 1080, "横長 (16:9)"),
        (600, 400, "バナー (3:2)")
    ]

    cols = st.columns(3)
    
    for i, (w, h, label) in enumerate(targets):
        # A. リサイズ
        resized_img = smart_resize(image, w, h)
        
        # B. 文字入れ (選択されたフォントを使用)
        final_img = add_text_to_image(
            resized_img, 
            text_input, 
            current_font_path, 
            font_size, 
            text_color, 
            text_position
        )
        
        # C. 表示とダウンロード
        with cols[i]:
            st.write(f"**{label}** ({w}x{h})")
            st.image(final_img, use_container_width=True)
            
            buf = io.BytesIO()
            final_img.save(buf, format="JPEG", quality=95)
            byte_im = buf.getvalue()
            
            st.download_button(
                label=f"📥 保存 ({w}x{h})",
                data=byte_im,
                file_name=f"resized_text_{w}x{h}.jpg",
                mime="image/jpeg"
            )
