import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import io
import os
import copy # 設定コピー用

# --- 1. 画像処理関数 ---

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

    # 文字サイズ計測（フチ含む）
    outline_width = settings['outline_width']
    try:
        bbox = draw.textbbox((0, 0), display_text, font=font, stroke_width=outline_width)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
    except AttributeError:
        text_w, text_h = draw.textsize(display_text, font=font)

    # --- 位置計算 ---
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

    # 微調整 (オフセット) の適用
    final_x = base_x + settings['offset_x']
    final_y = base_y + settings['offset_y']

    # --- 帯の描画 ---
    if settings['band_enabled']:
        bp = settings['band_padding']
        bx1, by1 = final_x - bp, final_y - bp
        bx2, by2 = final_x + text_w + bp, final_y + text_h + bp
        r, g, b = hex_to_rgb(settings['band_color'])
        band_fill = (r, g, b, int(255 * (settings['band_opacity'] / 100)))
        draw.rectangle([bx1, by1, bx2, by2], fill=band_fill)

    # --- 影 (Shadow) の描画 ---
    # 影は文字の下に描画
    if settings['shadow_enabled']:
        sx = final_x + settings['shadow_x']
        sy = final_y + settings['shadow_y']
        draw.text(
            (sx, sy),
            display_text,
            font=font,
            fill=settings['shadow_color'],
            align="center" if settings['is_vertical'] else "left"
        )

    # --- メイン文字の描画 ---
    draw.text(
        (final_x, final_y), 
        display_text, 
        font=font, 
        fill=settings['color'], 
        stroke_width=outline_width, 
        stroke_fill=settings['outline_color'],
        align="center" if settings['is_vertical'] else "left"
    )

    return Image.alpha_composite(img_rgba, txt_layer).convert("RGB")


# --- 2. UIコンポーネント関数 ---

def render_text_settings_ui(unique_key_prefix, available_fonts, FONT_DIR):
    """テキスト設定UIを表示し、設定辞書のリストを返す"""
    
    settings_list = []
    
    for i in range(3): # テキスト1〜3
        with st.expander(f"テキスト {i+1}", expanded=(i==0)):
            uid = f"{unique_key_prefix}_{i}"
            
            content = st.text_input("文字", value="", key=f"tx_{uid}")
            
            # --- 基本デザイン ---
            col1, col2 = st.columns(2)
            with col1:
                size_pct = st.slider("サイズ(%)", 1, 50, 10, key=f"sz_{uid}")
                color = st.color_picker("文字色", "#FFFFFF", key=f"cl_{uid}")
                pos = st.selectbox("基本位置", ["中央", "左上", "左下", "右上", "右下"], index=0, key=f"ps_{uid}")
            
            with col2:
                is_vert = st.checkbox("縦書き", False, key=f"vt_{uid}")
                font_path = None
                if available_fonts:
                    font_name = st.selectbox("フォント", available_fonts, key=f"ft_{uid}")
                    font_path = os.path.join(FONT_DIR, font_name)
                
                # 微調整
                st.caption("位置微調整(px)")
                c_ox, c_oy = st.columns(2)
                off_x = c_ox.number_input("Xズレ", value=0, step=5, key=f"ox_{uid}")
                off_y = c_oy.number_input("Yズレ", value=0, step=5, key=f"oy_{uid}")

            # --- 詳細設定タブ ---
            t_edge, t_shadow, t_band = st.tabs(["フチ", "影", "帯"])
            
            with t_edge:
                c1, c2 = st.columns(2)
                outline_w = c1.number_input("太さ", 0, 20, 2, key=f"ow_{uid}")
                outline_c = c2.color_picker("色", "#000000", key=f"oc_{uid}")
            
            with t_shadow:
                shadow_on = st.checkbox("影をつける", False, key=f"son_{uid}")
                if shadow_on:
                    c1, c2, c3 = st.columns(3)
                    s_x = c1.number_input("X", value=5, key=f"sx_{uid}")
                    s_y = c2.number_input("Y", value=5, key=f"sy_{uid}")
                    s_c = c3.color_picker("影色", "#333333", key=f"sc_{uid}")
                else:
                    s_x, s_y, s_c = 0, 0, "#000000"

            with t_band:
                band_on = st.checkbox("帯あり", False, key=f"bon_{uid}")
                if band_on:
                    c1, c2 = st.columns(2)
                    b_col = c1.color_picker("帯色", "#FF0000", key=f"bc_{uid}")
                    b_op = c2.slider("濃さ", 0, 100, 70, key=f"bop_{uid}")
                    b_pad = st.slider("余白", 0, 100, 10, key=f"bp_{uid}")
                else:
                    b_col, b_op, b_pad = "#000000", 0, 0

            # 設定を保存
            settings_list.append({
                "text": content,
                "size_percent": size_pct,
                "color": color,
                "position": pos,
                "offset_x": off_x,
                "offset_y": off_y,
                "is_vertical": is_vert,
                "font_path": font_path,
                "outline_width": outline_w,
                "outline_color": outline_c,
                "shadow_enabled": shadow_on,
                "shadow_x": s_x,
                "shadow_y": s_y,
                "shadow_color": s_c,
                "band_enabled": band_on,
                "band_color": b_col,
                "band_opacity": b_op,
                "band_padding": b_pad
            })
            
    return settings_list


# --- 3. アプリメイン処理 ---

st.set_page_config(page_title="プロ仕様リサイズ", layout="wide")
st.title("📷 AIリサイズ & プロ仕様デザイン")

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

st.sidebar.header("🎨 デザイン編集")
st.sidebar.info("基本はSquareで作って、他へコピー＆微調整するのがおすすめです。")

# タブ定義
tab_sq, tab_wd, tab_bn = st.sidebar.tabs(["Square (基準)", "Wide", "Banner"])

# --- 1. Square設定 (マスター) ---
with tab_sq:
    st.subheader("🔲 Square (1080x1080)")
    # Squareの設定UIを表示
    square_configs = render_text_settings_ui("sq", available_fonts, FONT_DIR)

# --- 2. Wide設定 ---
with tab_wd:
    st.subheader("📺 Wide (1920x1080)")
    use_sq_for_wd = st.checkbox("🔗 Squareの設定をコピー", value=True, key="sync_wd")
    
    if use_sq_for_wd:
        st.caption("Squareの設定が適用されています。変更したい場合はチェックを外してください。")
        wide_configs = copy.deepcopy(square_configs) # コピーを使用
    else:
        wide_configs = render_text_settings_ui("wd", available_fonts, FONT_DIR)

# --- 3. Banner設定 ---
with tab_bn:
    st.subheader("🏷️ Banner (600x400)")
    use_sq_for_bn = st.checkbox("🔗 Squareの設定をコピー", value=True, key="sync_bn")
    
    if use_sq_for_bn:
        st.caption("Squareの設定が適用されています。変更したい場合はチェックを外してください。")
        banner_configs = copy.deepcopy(square_configs) # コピーを使用
    else:
        banner_configs = render_text_settings_ui("bn", available_fonts, FONT_DIR)

# 設定を辞書にまとめる
all_format_configs = {
    "Square": square_configs,
    "Wide": wide_configs,
    "Banner": banner_configs
}


# --- 4. 画像生成処理 ---

uploaded_file = st.file_uploader("画像をアップロード", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="元の画像", width=400)
    st.divider()

    cols = st.columns(len(TARGET_SPECS))

    for idx, (w, h, label_key) in enumerate(TARGET_SPECS):
        # 1. リサイズ
        processed_img = smart_resize(image, w, h)

        # 2. 設定取得
        current_texts = all_format_configs[label_key]

        # 3. テキスト合成
        for settings in current_texts:
            if settings['text']: 
                processed_img = add_text_layer(processed_img, settings)

        # 4. 表示
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
