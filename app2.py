import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import io
import os

# --- 1. 画像処理関数 (変更なし) ---

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
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def add_text_layer(img, settings):
    """設定に基づいて文字レイヤーを合成する関数"""
    text = settings['text']
    if not text:
        return img

    img_rgba = img.convert("RGBA")
    txt_layer = Image.new("RGBA", img_rgba.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(txt_layer)
    W, H = img.size

    # フォントサイズ (画像の高さに対する％)
    font_size_px = int(H * (settings['size_percent'] / 100))
    font_size_px = max(10, font_size_px)

    # フォントロード
    font_path = settings['font_path']
    try:
        if font_path and os.path.exists(font_path):
            font = ImageFont.truetype(font_path, font_size_px)
        else:
            font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    display_text = text
    if settings['is_vertical']:
        display_text = "\n".join(list(text))

    outline_width = settings['outline_width']
    try:
        bbox = draw.textbbox((0, 0), display_text, font=font, stroke_width=outline_width)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
    except AttributeError:
        text_w, text_h = draw.textsize(display_text, font=font)

    # 位置計算
    position_preset = settings['position']
    padding = int(min(W, H) * 0.05)
    base_x, base_y = 0, 0

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

    # 帯の描画
    if settings['band_enabled']:
        bp = settings['band_padding']
        bx1, by1 = base_x - bp, base_y - bp
        bx2, by2 = base_x + text_w + bp, base_y + text_h + bp
        r, g, b = hex_to_rgb(settings['band_color'])
        band_fill = (r, g, b, int(255 * (settings['band_opacity'] / 100)))
        draw.rectangle([bx1, by1, bx2, by2], fill=band_fill)

    # 文字描画
    draw.text(
        (base_x, base_y), 
        display_text, 
        font=font, 
        fill=settings['color'], 
        stroke_width=outline_width, 
        stroke_fill=settings['outline_color'],
        align="center" if settings['is_vertical'] else "left"
    )

    return Image.alpha_composite(img_rgba, txt_layer).convert("RGB")


# --- 2. アプリ設定とUI ---

st.set_page_config(page_title="フォーマット別リサイズ", layout="wide")
st.title("📷 AIリサイズ & フォーマット別編集")

# フォント準備
FONT_DIR = "fonts"
available_fonts = []
if os.path.exists(FONT_DIR):
    available_fonts = [f for f in os.listdir(FONT_DIR) if f.endswith(('.ttf', '.otf'))]

# ターゲット定義
TARGET_SPECS = [
    (1080, 1080, "Square"),
    (1920, 1080, "Wide"),
    (600, 400, "Banner")
]

# --- サイドバー設定 (フォーマット単位のタブ) ---
st.sidebar.header("🎨 デザイン設定")

# 1. まずフォーマットごとにタブを作る
# リスト内包表記でタブオブジェクトのリストを作成
format_tabs = st.sidebar.tabs([label for _, _, label in TARGET_SPECS])

# 全フォーマットの設定を保存する辞書
# { "Square": [text1_conf, text2_conf...], "Wide": ... }
all_format_configs = {}

# zipを使って、スペック情報とタブを同時にループ処理
for (w, h, label), tab in zip(TARGET_SPECS, format_tabs):
    with tab:
        st.caption(f"📺 {label} ({w}x{h}) 用の設定")
        
        # このフォーマット用のテキスト設定リスト
        current_format_texts = []
        
        # テキスト1〜3の入力欄生成
        for i in range(3):
            # Expanderで閉じておく（すっきりさせるため）
            with st.expander(f"テキスト {i+1}", expanded=(i==0)):
                # キー(Key)をユニークにする必要がある：フォーマット名_番号_項目名
                uid = f"{label}_{i}"
                
                # 初期値は空文字 (value="")
                content = st.text_input("文字", value="", key=f"tx_{uid}")
                
                col1, col2 = st.columns(2)
                with col1:
                    size_pct = st.slider("サイズ(%)", 1, 50, 10, key=f"sz_{uid}")
                    color = st.color_picker("文字色", "#FFFFFF", key=f"cl_{uid}")
                    pos = st.selectbox("位置", ["中央", "左上", "左下", "右上", "右下"], index=0, key=f"ps_{uid}")
                
                with col2:
                    is_vert = st.checkbox("縦書き", False, key=f"vt_{uid}")
                    
                    font_path = None
                    if available_fonts:
                        font_name = st.selectbox("フォント", available_fonts, key=f"ft_{uid}")
                        font_path = os.path.join(FONT_DIR, font_name)
                    
                    outline_w = st.number_input("フチ太さ", 0, 10, 2, key=f"ow_{uid}")
                    outline_c = st.color_picker("フチ色", "#000000", key=f"oc_{uid}")

                # 帯設定
                b_on = st.checkbox("帯あり", False, key=f"bn_{uid}")
                if b_on:
                    b_col = st.color_picker("帯色", "#FF0000", key=f"bc_{uid}")
                    b_op = st.slider("濃さ", 0, 100, 70, key=f"bo_{uid}")
                    b_pad = st.slider("余白", 0, 50, 10, key=f"bp_{uid}")
                else:
                    b_col, b_op, b_pad = "#000000", 0, 0

                # 設定を保存
                current_format_texts.append({
                    "text": content,
                    "size_percent": size_pct,
                    "color": color,
                    "position": pos,
                    "is_vertical": is_vert,
                    "font_path": font_path,
                    "outline_width": outline_w,
                    "outline_color": outline_c,
                    "band_enabled": b_on,
                    "band_color": b_col,
                    "band_opacity": b_op,
                    "band_padding": b_pad
                })
        
        # 辞書に登録
        all_format_configs[label] = current_format_texts


# --- 3. メイン処理 ---

uploaded_file = st.file_uploader("画像をアップロード", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="元の画像", width=400)
    st.divider()

    cols = st.columns(len(TARGET_SPECS))

    for idx, (w, h, label_key) in enumerate(TARGET_SPECS):
        # 1. 画像のリサイズ
        processed_img = smart_resize(image, w, h)

        # 2. そのフォーマット用の設定を取得
        format_specific_texts = all_format_configs[label_key]

        # 3. テキストレイヤーを重ねる (入力があるものだけ)
        for settings in format_specific_texts:
            if settings['text']: # テキストが空でなければ描画
                processed_img = add_text_layer(processed_img, settings)

        # 4. 表示とダウンロード
        with cols[idx]:
            st.write(f"**{label_key}** ({w}x{h})")
            st.image(processed_img, use_container_width=True)

            buf = io.BytesIO()
            processed_img.save(buf, format="JPEG", quality=95)
            
            st.download_button(
                label="📥 保存",
                data=buf.getvalue(),
                file_name=f"{label_key}_{w}x{h}.jpg",
                mime="image/jpeg",
                key=f"dl_{idx}"
            )
