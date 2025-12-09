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

def hex_to_rgb(hex_color):
    """HEX色コードを(r, g, b)タプルに変換"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def add_text_advanced(img, text, font_path, font_size, text_color, 
                      is_vertical, outline_width, outline_color, 
                      band_enabled, band_color, band_opacity, band_padding,
                      position_preset, offset_x, offset_y):
    """高度な文字入れ関数"""
    if not text:
        return img

    # 画像をRGBAモードに変換（透過処理のため）
    img_rgba = img.convert("RGBA")
    # 文字描画用の透明レイヤーを作成
    txt_layer = Image.new("RGBA", img_rgba.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(txt_layer)
    W, H = img.size

    # フォント設定
    try:
        if font_path and os.path.exists(font_path):
            font = ImageFont.truetype(font_path, font_size)
        else:
            font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    # 縦書き対応（簡易版：改行を入れる）
    display_text = text
    if is_vertical:
        display_text = "\n".join(list(text))

    # テキストサイズ取得
    try:
        bbox = draw.textbbox((0, 0), display_text, font=font, stroke_width=outline_width)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
    except AttributeError:
        # 古いPillow用
        text_w, text_h = draw.textsize(display_text, font=font)

    # 基準位置の計算
    base_x, base_y = 0, 0
    padding = 20 # 画面端からの余裕

    if position_preset == "中央":
        base_x = (W - text_w) / 2
        base_y = (H - text_h) / 2
    elif position_preset == "左上":
        base_x = padding
        base_y = padding
    elif position_preset == "左下":
        base_x = padding
        base_y = H - text_h - padding
    elif position_preset == "右上":
        base_x = W - text_w - padding
        base_y = padding
    elif position_preset == "右下":
        base_x = W - text_w - padding
        base_y = H - text_h - padding

    # 微調整を適用
    final_x = base_x + offset_x
    final_y = base_y + offset_y

    # --- 帯（背景）の描画 ---
    if band_enabled:
        # 帯の座標計算（文字サイズ + 余白）
        # もし「画面幅いっぱいの帯」にしたい場合はここを調整しますが、今回は「文字の背景」とします
        bx1 = final_x - band_padding
        by1 = final_y - band_padding
        bx2 = final_x + text_w + band_padding
        by2 = final_y + text_h + band_padding
        
        # 帯の色設定 (RGBA)
        r, g, b = hex_to_rgb(band_color)
        band_fill = (r, g, b, int(255 * (band_opacity / 100)))
        
        # 帯を描画
        draw.rectangle([bx1, by1, bx2, by2], fill=band_fill)

    # --- 文字の描画 ---
    # 縁取り付きで描画
    draw.text(
        (final_x, final_y), 
        display_text, 
        font=font, 
        fill=text_color, 
        stroke_width=outline_width, 
        stroke_fill=outline_color,
        align="center" if is_vertical else "left"
    )

    # 元画像とテキストレイヤーを合成
    combined = Image.alpha_composite(img_rgba, txt_layer)
    return combined.convert("RGB") # JPEG保存用にRGBに戻す


# --- アプリのメイン処理 ---

st.set_page_config(page_title="高機能リサイズ＆文字入れ", layout="wide")
st.title("📷 AI自動リサイズ & プロ仕様文字入れ")

# --- サイドバー設定 ---

with st.sidebar:
    st.header("📝 テキスト入力")
    text_input = st.text_area("追加する文字", "Sale\n50% OFF", height=70)
    is_vertical = st.checkbox("縦書きモード (日本語推奨)")

    # フォント選択
    FONT_DIR = "fonts"
    current_font_path = None
    if os.path.exists(FONT_DIR):
        available_fonts = [f for f in os.listdir(FONT_DIR) if f.endswith(('.ttf', '.otf'))]
        if available_fonts:
            selected_font_name = st.selectbox("フォント選択", available_fonts)
            current_font_path = os.path.join(FONT_DIR, selected_font_name)
        else:
            st.warning("fontsフォルダにファイルがありません")
    
    st.divider()

    # タブで設定を整理
    tab1, tab2, tab3 = st.tabs(["🎨 デザイン", "🔲 帯・背景", "📐 配置・微調整"])

    with tab1:
        st.subheader("文字デザイン")
        font_size = st.slider("サイズ", 10, 200, 60)
        text_color = st.color_picker("文字色", "#FFFFFF")
        
        st.subheader("境界線 (フチ)")
        outline_width = st.slider("フチの太さ", 0, 10, 2)
        outline_color = st.color_picker("フチの色", "#000000")

    with tab2:
        st.subheader("背景の帯")
        band_enabled = st.toggle("文字の背景に帯をつける", value=False)
        band_color = st.color_picker("帯の色", "#FF0000")
        band_opacity = st.slider("帯の不透明度 (%)", 0, 100, 70)
        band_padding = st.slider("帯の広さ (パディング)", 0, 100, 20)

    with tab3:
        st.subheader("位置設定")
        position_preset = st.selectbox("基本位置", ["中央", "右下", "左下", "右上", "左上"], index=0)
        
        st.caption("微調整 (ピクセル)")
        col_x, col_y = st.columns(2)
        with col_x:
            offset_x = st.number_input("横方向 (X)", value=0, step=10)
        with col_y:
            offset_y = st.number_input("縦方向 (Y)", value=0, step=10)
            
# --- メイン画面処理 ---

uploaded_file = st.file_uploader("画像をアップロード", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="元の画像", width=400)
    st.divider()
    
    st.subheader("👇 仕上がりプレビュー")
    
    targets = [
        (1080, 1080, "正方形 (1:1)"),
        (1920, 1080, "横長 (16:9)"),
        (600, 400, "バナー (3:2)")
    ]

    cols = st.columns(3)
    
    for i, (w, h, label) in enumerate(targets):
        # 1. リサイズ
        resized_img = smart_resize(image, w, h)
        
        # 2. 高度な文字入れ
        final_img = add_text_advanced(
            resized_img, 
            text_input, 
            current_font_path, 
            font_size, 
            text_color,
            is_vertical,
            outline_width,
            outline_color,
            band_enabled,
            band_color,
            band_opacity,
            band_padding,
            position_preset,
            offset_x,
            offset_y
        )
        
        # 3. 表示とダウンロード
        with cols[i]:
            st.write(f"**{label}**")
            st.image(final_img, use_container_width=True)
            
            buf = io.BytesIO()
            final_img.save(buf, format="JPEG", quality=95)
            byte_im = buf.getvalue()
            
            st.download_button(
                label=f"📥 保存",
                data=byte_im,
                file_name=f"processed_{w}x{h}.jpg",
                mime="image/jpeg",
                key=f"dl_{i}"
            )
