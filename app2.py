import streamlit as st
from streamlit_cropper import st_cropper
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
import io
import os
import math

# --- 1. 画像処理関数 (文字入れなどはそのまま維持) ---

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def add_text_layer(img, settings):
    """設定に基づいて文字レイヤーを合成する関数"""
    text = settings['text']
    if not text:
        return img

    img = img.convert("RGBA")
    W, H = img.size

    # --- フォント準備 ---
    font_size_px = int(H * (settings['size_percent'] / 100))
    font_size_px = max(10, font_size_px)

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

    dummy_draw = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    outline_width = settings['outline_width']
    try:
        bbox = dummy_draw.textbbox((0, 0), display_text, font=font, stroke_width=outline_width)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
    except AttributeError:
        text_w, text_h = dummy_draw.textsize(display_text, font=font)

    # --- 基準位置計算 ---
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

    final_x = base_x + settings['offset_x']
    final_y = base_y + settings['offset_y']

    # --- 影レイヤー ---
    shadow_layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    if settings['shadow_enabled']:
        s_draw = ImageDraw.Draw(shadow_layer)
        angle_rad = math.radians(settings['shadow_angle'])
        s_off_x = settings['shadow_dist'] * math.cos(angle_rad)
        s_off_y = settings['shadow_dist'] * math.sin(angle_rad)
        sx = final_x + s_off_x
        sy = final_y + s_off_y
        
        s_draw.text((sx, sy), display_text, font=font, fill=settings['shadow_color'], align="center" if settings['is_vertical'] else "left")
        if settings['shadow_blur'] > 0:
            shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(settings['shadow_blur']))

    # --- 文字レイヤー ---
    text_layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    t_draw = ImageDraw.Draw(text_layer)

    if settings['band_enabled']:
        bp = settings['band_padding']
        bx1, by1 = final_x - bp, final_y - bp
        bx2, by2 = final_x + text_w + bp, final_y + text_h + bp
        r, g, b = hex_to_rgb(settings['band_color'])
        band_fill = (r, g, b, int(255 * (settings['band_opacity'] / 100)))
        t_draw.rectangle([bx1, by1, bx2, by2], fill=band_fill)

    t_draw.text(
        (final_x, final_y), display_text, font=font, fill=settings['color'], 
        stroke_width=outline_width, stroke_fill=settings['outline_color'],
        align="center" if settings['is_vertical'] else "left"
    )

    combined = Image.alpha_composite(img, shadow_layer)
    combined = Image.alpha_composite(combined, text_layer)
    return combined.convert("RGB")

# --- 2. UIコンポーネント関数 ---

def render_text_settings_ui(unique_key_prefix, available_fonts, FONT_DIR, defaults=None):
    """テキスト設定UI (省略せず全機能を維持)"""
    settings_list = []
    
    # 簡易化のため1つのテキストのみ設定可能にします（複数必要ならrange(3)に戻してください）
    for i in range(1): 
        def get_def(key, fallback):
            if defaults and i < len(defaults):
                return defaults[i].get(key, fallback)
            return fallback

        with st.expander(f"📝 テキスト設定", expanded=True):
            uid = f"{unique_key_prefix}_{i}"
            content = st.text_input("文字を入力", value=get_def('text', ""), key=f"tx_{uid}")
            
            col1, col2 = st.columns(2)
            with col1:
                size_pct = st.slider("サイズ", 1, 50, get_def('size_percent', 10), key=f"sz_{uid}")
                color = st.color_picker("文字色", get_def('color', "#FFFFFF"), key=f"cl_{uid}")
                pos_opts = ["中央", "左上", "左下", "右上", "右下"]
                default_pos = get_def('position', "中央")
                pos_idx = pos_opts.index(default_pos) if default_pos in pos_opts else 0
                pos = st.selectbox("配置", pos_opts, index=pos_idx, key=f"ps_{uid}")

            with col2:
                is_vert = st.checkbox("縦書き", get_def('is_vertical', False), key=f"vt_{uid}")
                font_idx = 0
                default_path = get_def('font_path', None)
                if default_path and available_fonts:
                    fname = os.path.basename(default_path)
                    if fname in available_fonts: font_idx = available_fonts.index(fname)
                
                font_name = "Default"
                font_path = None
                if available_fonts:
                    font_name = st.selectbox("フォント", available_fonts, index=font_idx, key=f"ft_{uid}")
                    font_path = os.path.join(FONT_DIR, font_name)
            
            # --- 装飾系 ---
            t_edge, t_shadow, t_band = st.tabs(["フチ", "影", "帯"])
            with t_edge:
                c1, c2 = st.columns(2)
                outline_w = c1.number_input("太さ", 0, 20, get_def('outline_width', 2), key=f"ow_{uid}")
                outline_c = c2.color_picker("色", get_def('outline_color', "#000000"), key=f"oc_{uid}")
            with t_shadow:
                shadow_on = st.checkbox("影", get_def('shadow_enabled', False), key=f"son_{uid}")
                s_ang, s_dist, s_blur, s_c = 45, 10, 5, "#333333"
                if shadow_on:
                    s_dist = st.slider("距離", 0, 50, get_def('shadow_dist', 10), key=f"sd_{uid}")
                    s_c = st.color_picker("影色", get_def('shadow_color', "#333333"), key=f"sc_{uid}")
            with t_band:
                band_on = st.checkbox("帯", get_def('band_enabled', False), key=f"bon_{uid}")
                b_col, b_op, b_pad = "#FF0000", 70, 10
                if band_on:
                    b_col = st.color_picker("帯色", get_def('band_color', "#FF0000"), key=f"bc_{uid}")
                    b_op = st.slider("透明度", 0, 100, get_def('band_opacity', 70), key=f"bop_{uid}")

            settings_list.append({
                "text": content, "size_percent": size_pct, "color": color, "position": pos,
                "offset_x": 0, "offset_y": 0, "is_vertical": is_vert, "font_path": font_path,
                "outline_width": outline_w, "outline_color": outline_c,
                "shadow_enabled": shadow_on, "shadow_angle": 45, "shadow_dist": s_dist,
                "shadow_blur": 5, "shadow_color": s_c,
                "band_enabled": band_on, "band_color": b_col, "band_opacity": b_op, "band_padding": 10
            })
    return settings_list

# --- 3. メインアプリ ---

st.set_page_config(page_title="直感的リサイズApp", layout="wide")
st.title("✂️ 直感的操作 & デザイン")

FONT_DIR = "fonts"
available_fonts = []
if os.path.exists(FONT_DIR):
    available_fonts = [f for f in os.listdir(FONT_DIR) if f.endswith(('.ttf', '.otf'))]

# ターゲットサイズの定義
TARGETS = {
    "Square": (1080, 1080),
    "Wide": (1200, 628),
    "Banner": (600, 400)
}

st.markdown("""
<style>
    /* クロップ画面の余白調整 */
    .block-container {padding-top: 2rem;}
</style>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("画像をアップロードしてください", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    # 画像読み込み
    original_image = Image.open(uploaded_file)
    if original_image.mode != "RGB":
        original_image = original_image.convert("RGB")

    st.write("---")
    
    # タブでサイズ切り替え
    tab_sq, tab_wd, tab_bn = st.tabs(["🔲 Square (1080px)", "📺 Wide (1200x628)", "🏷️ Banner (600x400)"])
    
    tabs = zip([tab_sq, tab_wd, tab_bn], TARGETS.items())

    for tab, (label, (w, h)) in tabs:
        with tab:
            col_edit, col_preview = st.columns([1, 1])
            
            # --- 左カラム：直感的な位置調整（クロッパー） ---
            with col_edit:
                st.subheader("1. 位置と範囲を決める")
                st.info("👇 下の画像の「枠」を動かして、切り取る範囲を決めてください")
                
                # アスペクト比を計算
                aspect_ratio = (w, h)
                
                # 直感的クロップUI
                cropped_img_preview = st_cropper(
                    original_image,
                    realtime_update=True,
                    box_color='#0000FF', # 青い枠
                    aspect_ratio=aspect_ratio,
                    key=f"cropper_{label}"
                )
                
                # テキスト設定UI
                st.subheader("2. 文字を入れる")
                text_configs = render_text_settings_ui(label, available_fonts, FONT_DIR)

            # --- 右カラム：仕上がり確認 ---
            with col_preview:
                st.subheader("3. 仕上がり確認")
                
                if cropped_img_preview:
                    # クロップされた画像を、最終出力サイズへリサイズ（高画質補間）
                    final_img = cropped_img_preview.resize((w, h), Image.LANCZOS)
                    
                    # 青背景キャンバスなどは不要（クロップ＝画面いっぱいに広げる挙動のため）
                    # 文字合成
                    for settings in text_configs:
                        final_img = add_text_layer(final_img, settings)

                    # 表示
                    st.image(final_img, caption=f"{label} ({w}x{h})", use_container_width=True)
                    
                    # ダウンロードボタン
                    buf = io.BytesIO()
                    final_img.save(buf, format="JPEG", quality=95)
                    st.download_button(
                        label=f"📥 {label}画像を保存",
                        data=buf.getvalue(),
                        file_name=f"{label}_{w}x{h}.jpg",
                        mime="image/jpeg",
                        key=f"dl_{label}"
                    )
